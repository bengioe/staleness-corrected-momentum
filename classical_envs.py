import math
import numpy as np
import os
import gym
import pickle
import gzip
import torch
import torch.nn as nn

gym.logger.set_level(gym.logger.ERROR)

class MountainCarEnv:

    def __init__(self, seed=142739):
        self._env = gym.make("MountainCar-v0")
        self._env.seed(seed)

    def step(self, action):
        s, r, t, i = self._env.step(action)
        r *= 0.1
        return s,r,t,i

    def reset(self):
        self._env.reset()
        self._env.unwrapped.state = np.array([
            self._env.np_random.uniform(low=self._env.min_position, high=self._env.max_position),
            self._env.np_random.uniform(low=-self._env.max_speed, high=self._env.max_speed),])
        return np.array(self._env.state)

    def optimal_action(self):
        if self._env.unwrapped.state[1] < 0:
            return 0
        else:
            return 2

    @classmethod
    def get_oracle(cls, create=True):
        return _get_oracle(create, MountainCarEnv, "mc_oracle")


class AcrobotEnv:
    expert = None

    def __init__(self, seed=142739):
        self._env = gym.make("Acrobot-v1")
        self._env.seed(seed)

    def step(self, action):
        s, r, t, i = self._env.step(action)
        r *= 0.1
        return s,r,t,i

    def o_reset(self):
        return self._env.reset()

    def reset(self):
        return self._env.reset()

    def optimal_action(self): # Added
        if AcrobotEnv.expert is None:
            AcrobotEnv.expert = pickle.load(gzip.open(f'oracles/acrobot_optimal_policy.pkl.gz','rb'))
        s = torch.tensor(self._env.unwrapped._get_ob()).float().unsqueeze(0)
        return AcrobotEnv.expert(s).squeeze(0).argmax()

    @classmethod
    def get_oracle(cls, create=True):
        return _get_oracle(create, AcrobotEnv, "acrobot_oracle", train_steps=50000, full_traj=True)

    @classmethod
    def train_optimal_policy(cls, gamma=0.99):
        # Simple DQN implementation
        import torch
        import torch.nn as nn
        import copy
        env = AcrobotEnv(42)
        test_env = AcrobotEnv(43)
        test_env2 = AcrobotEnv(42)
        nact = env._env.action_space.n
        nhid = 64
        xdim = env._env.observation_space.shape[0]
        act = nn.LeakyReLU()
        oracle = nn.Sequential(nn.Linear(xdim, nhid), act,
                               nn.Linear(nhid, nhid), act,
                               nn.Linear(nhid, nact))

        target = copy.deepcopy(oracle)
        opt = torch.optim.Adam(oracle.parameters(), 1e-3)
        tf = lambda x: torch.tensor(x).float()
        best_model = None
        best_score = 10000

        buf = []

        s = env.o_reset()
        u = 0

        for i in range(100000):
            eps = 0.95 * max(1 - i / 10000, 0) + 0.05
            a = (oracle(tf(s).unsqueeze(0)).squeeze(0).argmax()
                 if np.random.uniform(0, 1) < (1-eps) else np.random.randint(nact))
            sp, r, t, _ = env.step(a)
            buf.append((s, a, sp, r, env._env.unwrapped._terminal()))
            s = sp if not t else env.o_reset()
            if len(buf) < 1000:
                continue

            transitions = [buf[i] for i in np.random.randint(0, len(buf), 128)]
            st, at, stp, r, done = [tf([i[j] for i in transitions])
                                    for j in range(5)]
            vt = oracle(st).gather(1, at.long().unsqueeze(1)).squeeze(1)
            vtp = target(stp).max(1).values.detach()
            loss = (vt - (r + gamma * vtp * (1-done))).pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

            if i and not i % 2000:
                target = copy.deepcopy(oracle)

            if not i % 1000:
                scores = []
                for k in range(50):
                    ts = test_env.o_reset()
                    for j in range(1000):
                        ts, r, done, _ = test_env.step(
                            oracle(tf(ts).unsqueeze(0)).squeeze(0).argmax()
                            if np.random.uniform(0, 1) < 0.95 else np.random.randint(nact))
                        if done:
                            scores.append(j)
                            break

                score = np.mean(scores)
                if score < best_score:
                    best_model = copy.deepcopy(oracle)
                    best_score = score
                print(i, score, best_score, loss.item(), vt.min().item(), vt.mean().item(), vt.max().item())
        pickle.dump(best_model, gzip.open(f'oracles/acrobot_optimal_policy.pkl.gz','wb'))





class CartpoleEnv:
    expert = None

    def __init__(self, seed=142739):
        self._env = gym.make("CartPole-v1")
        self._env.seed(seed)

        self._env.observation_space.shape = (5,)

    def f(self, s):
        return np.concatenate([s, [self._env._elapsed_steps / 500]])

    def step(self, action):
        s, r, t, i = self._env.step(action)
        r *= 0.1
        return self.f(s),r,t,i

    def true_done(self):
        x, _, theta, _ = self._env.state
        return bool(
            x < -self._env.x_threshold
            or x > self._env.x_threshold
            or theta < -self._env.theta_threshold_radians
            or theta > self._env.theta_threshold_radians
        )

    def o_reset(self):
        return self.f(self._env.reset())

    def reset(self):
        return self.f(self._env.reset())

    def optimal_action(self): # Added
        if CartpoleEnv.expert is None:
            # from https://github.com/araffin/rl-baselines-zoo/tree/master/trained_agents
            params = pickle.load(gzip.open(f'oracles/cartpole_optimal_policy.pkl.gz','rb'))
            expert = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, 2))
            with torch.no_grad():
                [i.set_(torch.tensor(j.T if j.ndim == 2 else j).float())
                 for i, j in zip(expert.parameters(), params)]
            CartpoleEnv.expert = expert
        if np.random.uniform(0,1) < 0.8:
            s = torch.tensor(self._env.state).float().unsqueeze(0)
            return CartpoleEnv.expert(s).squeeze(0).argmax().item()
        return np.random.randint(0,2)

    @classmethod
    def get_oracle(cls, create=True):
        return _get_oracle(create, CartpoleEnv, "cartpole_oracle", train_steps=5000, full_traj=True)

    @classmethod
    def print_stat_dist(cls):
        import tqdm
        import scipy.stats
        env = CartpoleEnv()
        s = env.o_reset()
        all_s = []
        for i in tqdm.tqdm(range(100000)):
            s, r, t, _ = env.step(env.optimal_action())
            if t:
                s = env.o_reset()
            all_s.append(s)
        g = scipy.stats.gaussian_kde(np.float32(all_s).T)
        pickle.dump(g, gzip.open('oracles/cartpole_kde.pkl.gz', 'wb'))

class LunarLanderEnv:
    expert = None

    def __init__(self, seed=142739):
        self._env = gym.make("LunarLander-v2")
        self._env.seed(seed)
        self._env._max_episode_steps = 2000

    def step(self, action):
        s, r, t, i = self._env.step(action)
        r *= 0.1
        self.last_s = s
        return s,r,t,i

    def o_reset(self):
        self.last_s = self._env.reset()
        return self.last_s

    def reset(self):
        self.last_s = self._env.reset()
        return self.last_s

    def optimal_action(self): # Added
        if LunarLanderEnv.expert is None:
            # from https://github.com/araffin/rl-baselines-zoo/tree/master/trained_agents
            params = pickle.load(gzip.open(f'oracles/lunarlander_optimal_policy.pkl.gz','rb'))
            expert = nn.Sequential(nn.Linear(8, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, 4))
            with torch.no_grad():
                [i.set_(torch.tensor(j.T if j.ndim == 2 else j).float())
                 for i, j in zip(expert.parameters(), params)]
            LunarLanderEnv.expert = expert
        if 1 or np.random.uniform(0,1) < 0.95:
            s = torch.tensor(self.last_s).float().unsqueeze(0)
            return CartpoleEnv.expert(s).squeeze(0).argmax().item()
        return np.random.randint(0,4)

    @classmethod
    def get_oracle(cls, create=True):
        return _get_oracle(create, LunarLanderEnv,
                           "lunarlander_oracle", train_steps=50000,
                           full_traj=True, nhid=256, lr=1e-4, mbsize=2048)



def _get_oracle(create=True, env=MountainCarEnv, path="mc_oracle",
                gamma=0.99, train_steps=5000, full_traj=False,
                nhid=64, lr=1e-3, mbsize=128):
    import os.path
    import torch
    import torch.nn as nn
    import tqdm

    if os.path.exists(f'oracles/{path}.pkl.gz'):
        oracle = pickle.load(gzip.open(f'oracles/{path}.pkl.gz','rb'))
    else:
        if not create:
            raise ValueError()
        dataset = []

        env = env()
        if full_traj:
            s = env.reset()
            ss = []
            rs = []
            for i in tqdm.tqdm(range(100000)):
                ss.append(s)
                s,r,done,_ = env.step(env.optimal_action())
                rs.append(r)
                if done:
                    gs = [(np.float32(rs[i:]) * (gamma ** np.arange(len(rs)-i))).sum()
                          for i in range(len(rs))]
                    data = list(zip(ss, gs))
                    dataset += data
                    ss = []
                    rs = []
                    s = env.reset()
        else:
            for i in range(10000):

                s0 = env.reset()
                g = 0
                done = False
                t = 0
                while not done:
                    s,r,done,_ = env.step(env.optimal_action())
                    g += r * gamma ** t
                    t += 1
                    if t > 2/(1-gamma):
                        break
                if not i % 50:
                    print(t, int(2/(1-gamma)))
                dataset.append((s0, g))

        act = nn.LeakyReLU()
        oracle = nn.Sequential(nn.Linear(env._env.observation_space.shape[0], nhid), act,
                               nn.Linear(nhid, nhid), act,
                               nn.Linear(nhid, nhid), act,
                               nn.Linear(nhid, nhid), act,
                               nn.Linear(nhid, 1))
        opt = torch.optim.Adam(oracle.parameters(), lr)
        losses = []
        _x = torch.tensor([i[0] for i in dataset]).float()
        _y = torch.tensor([i[1] for i in dataset]).float()
        for i in range(train_steps):
            idx = np.random.randint(0, len(dataset), mbsize)
            x = _x[idx]
            y = _y[idx]
            o = oracle(x).squeeze(1)
            loss = (o - y).pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            if not i % 100:
                print(i, loss.item(), y.mean().item())
        if create and not os.path.exists(f'oracles/{path}.pkl.gz'):
            pickle.dump(oracle, gzip.open(f'oracles/{path}.pkl.gz','wb'))
    return oracle




if __name__ == '__main__':

    if not os.path.exists('oracles/acrobot_optimal_policy.pkl.gz'):
        AcrobotEnv.train_optimal_policy()
    if not os.path.exists('oracles/acrobot_oracle.pkl.gz'):
        AcrobotEnv.get_oracle()

    if not os.path.exists('oracles/cartpole_kde.pkl.gz'):
        CartpoleEnv.print_stat_dist()
    if not os.path.exists('oracles/cartpole_oracle.pkl.gz'):
        CartpoleEnv.get_oracle()

    if not os.path.exists('oracles/lunarlander_oracle.pkl.gz'):
        LunarLanderEnv.get_oracle()
