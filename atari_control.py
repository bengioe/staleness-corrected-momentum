import argparse

import math
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

import copy
import gc
import pickle
import lmdb
import gzip

import neural_network as mm
from neural_network import tf, tint
from replay_buffer import ReplayBufferV2
from atari_env import AtariEnv
from rainbow import DQN

from scmsgd import SCMSGD, OptChain, SCMTDProp


parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--nhid", default=32, help="#hidden", type=int)
parser.add_argument("--buffer_size", default=100000, help="Replay buffer size",type=int)
parser.add_argument("--num_iterations", default=500_000, type=int)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--head_type", default='normal', help='normal or slim or none')
parser.add_argument("--body_type", default='normal', help='normal or slim or tiny')
parser.add_argument("--act", default='lrelu')
parser.add_argument("--td_n_step", default=1, type=int)
parser.add_argument("--num_env_steps", default=1, type=int)
parser.add_argument("--opt", default='adam')
parser.add_argument("--opt_momentum", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.999, type=float)
parser.add_argument("--opt_diagonal", default=True, type=int)
parser.add_argument("--opt_block_diagonal", default=0, type=int)
parser.add_argument("--opt_ignore_vprime", default=0, type=int)
parser.add_argument("--opt_correct_adam", default=0, type=int)
parser.add_argument("--num_full_corr", default=0, type=int,
                    help="End parameters for which the full correction matrix (instead of diagonal) is computed")
parser.add_argument("--num_bot_corr", default=0, type=int,
                    help="Bottom parameters for which the full correction matrix (instead of diagonal) is computed")
parser.add_argument("--target_type", default='none') # frozen, none, moving
parser.add_argument("--target_clone_interval", default=10000, type=int)
parser.add_argument("--target_tau", default=0.01, type=float)
parser.add_argument("--checkpoint_freq", default=10000, type=int)
parser.add_argument("--test_freq", default=5000, type=int)
parser.add_argument("--env_name", default='ms_pacman')
parser.add_argument("--device", default='cuda', help="device")
parser.add_argument("--progress", default=False, action='store_true', help='display a progress bar')
parser.add_argument("--array", default='')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--test_run", default=False, action='store_true')
parser.add_argument("--save_path", default='results/')
parser.add_argument("--save_parameters", default=False, type=bool)
parser.add_argument("--measure_drift", default=False, type=bool)




class odict(dict):

  def __getattr__(self, x):
    return self[x]


def sl1(a, b):
  """Smooth L1 distance"""
  d = a - b
  u = abs(d)
  s = d**2
  m = (u < s).float()
  return u * m + s * (1 - m)


def make_opt(args, theta):
  theta = list(theta)
  if args.opt == "sgd":
    return torch.optim.SGD(theta, args.learning_rate, weight_decay=args.weight_decay)
  elif args.opt == "msgd":
    return torch.optim.SGD(theta, args.learning_rate, weight_decay=args.weight_decay,
                           momentum=args.opt_momentum,
                           dampening=args.opt_momentum)
  elif args.opt == "msgd_corr":
    if not args.opt_diagonal:
      print("args.opt_diagonal is False; you probably",
            "don't want the full covariance matrix for an atari model!",
            "Unless you have 22 Terabytes of RAM and a good book.")
    if args.num_full_corr > 0 or args.num_bot_corr == 0:
        # This chains two optimizers, one with the early (lower layers)
        # parameters, and the second with the last num_full_corr
        # parameters (presumably the parameters close to the value
        # prediction).  The first one is diagonal, while the second
        # maintains the full matrix (which would be way too large for the
        # entire model). To make this tractable, it is recommended to use
        # head_type == 'slim', which adds a slimmer head with fewer
        # parameters to maintain an O(n^2) matrix of.
        return OptChain([
          SCMTDProp(sub_theta, args.learning_rate, weight_decay=args.weight_decay,
                    momentum=args.opt_momentum,
                    dampening=args.opt_momentum,
                    diagonal=diag,
                    block_diagonal=args.block_diagonal,
                    #correct_adam=args.opt_correct_adam,
                    beta2=args.opt_beta2)
          for sub_theta, diag in ([(theta[:-args.num_full_corr], args.opt_diagonal),
                                   (theta[-args.num_full_corr:], False)] # Force non-diagonal
                                  if args.num_full_corr > 0 else
                                  [(theta, args.opt_diagonal)])])
    elif args.num_bot_corr > 0:
        return OptChain([
          SCMSGD(sub_theta, args.learning_rate, weight_decay=args.weight_decay,
                   momentum=args.opt_momentum,
                   dampening=args.opt_momentum,
                   diagonal=diag,
                   correct_adam=args.opt_correct_adam,
                   beta2=args.opt_beta2)
          for sub_theta, diag in [(theta[:args.num_bot_corr], False), # Force non-diagonal
                                  (theta[args.num_bot_corr:], args.opt_diagonal)]])
    elif args.num_bot_corr > 0 and False:
      return OptChain([
        SCMSGD(theta[:args.num_bot_corr], args.learning_rate, weight_decay=args.weight_decay,
                 momentum=args.opt_momentum,
                 dampening=args.opt_momentum,
                 diagonal=False,
                 correct_adam=args.opt_correct_adam,
                 beta2=args.opt_beta2),
        torch.optim.Adam(theta[args.num_bot_corr:], args.learning_rate, weight_decay=args.weight_decay,
                         betas=(args.opt_momentum, args.opt_beta2))])
  elif args.opt == "rmsprop":
    return torch.optim.RMSprop(theta, args.learning_rate, weight_decay=args.weight_decay)
  elif args.opt == "adam":
    return torch.optim.Adam(theta, args.learning_rate, weight_decay=args.weight_decay,
                            betas=(args.opt_momentum, args.opt_beta2))
  else:
    raise ValueError(args.opt)


def init_weights(m):
  if isinstance(m, torch.nn.Linear):
    k = np.sqrt(6 / (np.sum(m.weight.shape)))
    m.weight.data.uniform_(-k, k)
    m.bias.data.fill_(0)
  if isinstance(m, torch.nn.Conv2d):
    u,v,w,h = m.weight.shape
    k = np.sqrt(6 / (u + w*h*v))
    m.weight.data.uniform_(-k, k)
    m.bias.data.fill_(0)

def packobj(x):
  return gzip.compress(pickle.dumps(x))

def unpackobj(x):
  return pickle.loads(gzip.decompress(x))

def main(args):
  device = torch.device(args.device)
  mm.set_device(device)
  results_conn = lmdb.open(f'{args.save_path}/run_{args.run}', map_size=int(16 * 2 ** 30))
  params_conn = lmdb.open(f'{args.save_path}/run_{args.run}/params', map_size=int(16 * 2 ** 30))
  with results_conn.begin(write=True) as txn:
    txn.put(b'args', packobj(args))

  print(args)
  seed = args.run + 1_642_559  # A large prime number
  torch.manual_seed(seed)
  np.random.seed(seed)
  rng = np.random.RandomState(seed)
  env = AtariEnv(args.env_name)
  mbsize = args.mbsize
  nhid = args.nhid
  gamma = 0.99
  env.r_gamma = gamma
  checkpoint_freq = args.checkpoint_freq
  test_freq = args.test_freq
  target_tau = args.target_tau
  target_clone_interval = args.target_clone_interval
  target_type = args.target_type
  num_iterations = args.num_iterations
  num_Q_outputs = env.num_actions
  td_steps = args.td_n_step
  num_env_steps = args.num_env_steps
  measure_drift = args.measure_drift
  # Model

  act = {'lrelu': torch.nn.LeakyReLU(),
         'tanh': torch.nn.Tanh(),
         'elu': torch.nn.ELU(),
  }[args.act]
  # Body
  if args.body_type == 'normal':
    body = torch.nn.Sequential(torch.nn.Conv2d(4, nhid, 8, stride=4, padding=4), act,
                               torch.nn.Conv2d(nhid, nhid*2, 4, stride=2,padding=2), act,
                               torch.nn.Conv2d(nhid*2, nhid*2, 3,padding=1), act,
                               torch.nn.Flatten(),
                               torch.nn.Linear(nhid*2*12*12, nhid*16), act)
  elif args.body_type == 'slim_bot2':
    body = torch.nn.Sequential(torch.nn.Conv2d(4, nhid // 2, 8, stride=4, padding=4), act,
                               torch.nn.Conv2d(nhid // 2, nhid, 4, stride=2,padding=2), act,
                               torch.nn.Conv2d(nhid, nhid*2, 3,padding=1), act,
                               torch.nn.Flatten(),
                               torch.nn.Linear(nhid*2*12*12, nhid*16), act)
  elif args.body_type == 'added_bot3':
    body = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3, padding=1), act,
                               torch.nn.Conv2d(4, 4, 3, padding=1), act,
                               torch.nn.Conv2d(4, 4, 3, padding=1), act,
                               torch.nn.Conv2d(4, nhid, 8, stride=4, padding=4), act,
                               torch.nn.Conv2d(nhid, nhid*2, 4, stride=2,padding=2), act,
                               torch.nn.Conv2d(nhid*2, nhid*2, 3,padding=1), act,
                               torch.nn.Flatten(),
                               torch.nn.Linear(nhid*2*12*12, nhid*16), act)
  elif args.body_type == 'added_bot3A':
    body = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3, padding=1), act,
                               torch.nn.Conv2d(8, 8, 3, padding=1), act,
                               torch.nn.Conv2d(8, 8, 3, padding=1), act,
                               torch.nn.Conv2d(8, nhid, 8, stride=4, padding=4), act,
                               torch.nn.Conv2d(nhid, nhid*2, 4, stride=2,padding=2), act,
                               torch.nn.Conv2d(nhid*2, nhid*2, 3,padding=1), act,
                               torch.nn.Flatten(),
                               torch.nn.Linear(nhid*2*12*12, nhid*16), act)
  elif args.body_type == 'tiny':
    body = torch.nn.Sequential(torch.nn.Conv2d(4, nhid, 3, stride=2, padding=1), act, # 42
                               torch.nn.Conv2d(nhid, nhid, 3, stride=2, padding=1), act, # 21
                               torch.nn.Conv2d(nhid, nhid, 3, stride=2, padding=1), act, # 11
                               torch.nn.Conv2d(nhid, nhid, 3, stride=2, padding=1), act, # 6
                               torch.nn.Conv2d(nhid, num_Q_outputs, 6), # 1
                               torch.nn.Flatten())
  # Head
  if args.head_type == 'normal':
    head = torch.nn.Sequential(torch.nn.Linear(nhid*16, num_Q_outputs))
  elif args.head_type == 'slim': # Slim end to we can do block diagonal zeta
    head = torch.nn.Sequential(torch.nn.Linear(nhid*16, nhid), act,
                               torch.nn.Linear(nhid, nhid), act,
                               torch.nn.Linear(nhid, num_Q_outputs))
  elif args.head_type == 'slim2':
    head = torch.nn.Sequential(torch.nn.Linear(nhid*16, nhid * 2), act,
                               torch.nn.Linear(nhid * 2, num_Q_outputs))
  elif args.head_type == 'none':
    head = torch.nn.Sequential()

  Qf = torch.nn.Sequential(body, head)

  Qf.to(device)
  Qf.apply(init_weights)
  if args.target_type == 'none':
    Qf_target = Qf
  else:
    Qf_target = copy.deepcopy(Qf)

  opt = make_opt(args, Qf.parameters())
  opt.epsilon = 1e-2
  do_specific_backward = args.opt == 'msgd_corr'

  # Replay Buffer
  replay_buffer = ReplayBufferV2(seed, args.buffer_size)

  ar = lambda x: torch.arange(x.shape[0], device=x.device)

  losses = []
  num_iterations = 1 + num_iterations
  ignore_vprime = bool(args.opt_ignore_vprime)
  total_reward = 0
  last_end = 0
  last_rewards = []
  num_exploration_steps = 50_000
  final_epsilon = 0.05
  recent_states = []
  recent_values = []

  obs = env.reset()
  drift = 0

  # Run policy evaluation
  _prof = (tqdm(range(num_iterations), smoothing=0.001) if args.progress else range(num_iterations))
  for it in _prof:

    for eit in range(num_env_steps):
        if it < num_exploration_steps:
          epsilon = 1 - (it / num_exploration_steps) * (1 - final_epsilon)
        else:
          epsilon = final_epsilon

        if rng.uniform(0, 1) < epsilon:
          action = rng.randint(0, num_Q_outputs)
        else:
          with torch.no_grad():
            action = Qf(torch.tensor(obs / 255.0, device=device).unsqueeze(0).float()).argmax().item()

        obsp, r, done, info = env.step(action)
        total_reward += r
        replay_buffer.add(obs, action, r, done, env.enumber % 2)

        obs = obsp
        if done:
          obs = env.reset()
          with results_conn.begin(write=True) as txn:
            txn.put(f'episode_{env.enumber-1}'.encode(), packobj({
              "end": it,
              "start": last_end,
              "total_reward": total_reward
            }))
          last_end = it
          last_rewards = [total_reward] + last_rewards[:10]
          if args.progress:
            _prof.set_description_str(f'reward {int(100*total_reward)}, '
                                      f'{int(100*np.mean(last_rewards))}, '
                                      f'{drift:.5f}')
          total_reward = 0

    if replay_buffer.current_size < 5000:
      continue

    sample = replay_buffer.sample(mbsize, n_step=td_steps)

    if Qf_target is Qf:
      q = Qf(torch.cat([sample.s, sample.sp], 0))
      v = q[ar(sample.s), sample.a.long()]
      vp = q[ar(sample.sp)+sample.s.shape[0], sample.ap.long()]
    else:
      q = Qf(sample.s)
      v = q[ar(q), sample.a.long()]
      vp = Qf_target(sample.sp)[ar(q), sample.ap.long()]

    gamma_mask = (1 - sample.t.float()) * (gamma ** td_steps)
    target = sample.r + gamma_mask * vp
    loss = (v - target.detach()).pow(2)
    if do_specific_backward:
      opt.backward_and_step(v, vp, v - target, gamma_mask)
      #opt.set_predictions(v.mean(), gvp.mean() if not ignore_vprime else None)
    else:
      loss = loss.mean()
      loss.backward()
      opt.step()
      opt.zero_grad()

    losses.append(loss.item())

    if measure_drift:
      recent_states.append((sample.sp, sample.ap.long()))
      recent_values.append(vp.detach())

    if len(recent_states) >= 32:
      rs = torch.cat([i[0] for i in recent_states])
      ra = torch.cat([i[1] for i in recent_states])
      rvp = torch.cat(recent_values)
      with torch.no_grad():
        nvp = Qf_target(rs)[ar(ra), ra]

      drift = abs(rvp - nvp).mean().item()
      with results_conn.begin(write=True) as txn:
        txn.put(f'value_drift_{it}'.encode(), packobj(drift))
      recent_states = []
      recent_values = []


    if target_type == 'frozen' and it % target_clone_interval == 0:
      Qf_target = copy.deepcopy(Qf)
    elif target_type == 'moving':
      for target_param, param in zip(Qf_target.parameters(), Qf.parameters()):
        target_param.data.mul_(1-target_tau).add_(param, alpha=target_tau)

    if it % checkpoint_freq == 0 and args.save_parameters:
      with params_conn.begin(write=True) as txn:
        txn.put(f'parameters_last'.encode(), packobj(Qf.state_dict()))

    if it % test_freq == 0:
      mc_loss = 0
      n = 0
      with torch.no_grad():
        #print('|W|^2 =', sum([i.pow(2).sum() for i in Qf.parameters()]))
        #for sample in replay_buffer.iterate(512):
        while True:
          sample = replay_buffer.sample(512)
          n += sample.s.shape[0]
          q = Qf(sample.s).max(1).values
          mc_loss += (q - sample.g).pow(2).sum().item()
          if n > 10000:
            break
      with results_conn.begin(write=True) as txn:
        txn.put(f'mc-loss_{it}'.encode(), packobj((mc_loss/n,)))
        if it > 0:
          txn.put(f'train-loss_{it}'.encode(), packobj(losses))
        losses = []


    if np.isnan(loss.item()):
      print("Learning has diverged, nan loss")
      with results_conn.begin(write=True) as txn:
        txn.put(b'diverged', b'True')
      break
  print("Done.")


def array_sep_20(args):
  base = {
    'buffer_size': 100_000,
    'num_iterations': 200_000,
    'test_freq': 10000,
    'mbsize': 32,
    'learning_rate': 1e-4,
    'target_type': 'none',
    'head_type': 'none',
    'body_type': 'tiny',
  }
  all_hps = ([
    {**base,
     'opt_momentum': opt_momentum,
     'opt': opt,
     'opt_diagonal': opt_diagonal,
     'nhid': nhid,
     'td_n_step': n_step,
     }
    for seed in range(5)
    for learning_rate in [1e-4]
    for opt_momentum in [0.9, 0.99]
    for opt in ['msgd_corr', 'adam']
    for opt_diagonal in ([0,1] if opt == 'msgd_corr' else [1])
    for nhid in [16, 32]
    for n_step in [1, 5]
    for env in ['ms_pacman', 'breakout', 'qbert']
  ])
  return all_hps

def array_sep_22(args):
  base = {
    'buffer_size': 100_000,
    'num_iterations': 500_000,
    'test_freq': 20000,
    'mbsize': 32,
    'learning_rate': 1e-4,
    'target_type': 'none',
    'head_type': 'none',
    'body_type': 'tiny',
    'nhid': 16,
    'env_name': 'ms_pacman',
    'td_n_step': 5,
  }
  rng = np.random.default_rng(32)
  nrun = 40
  all_hps = ([
    {**base,
     'opt_momentum': opt_momentum,
     'opt': opt,
     'opt_diagonal': opt_diagonal,
     'opt_beta2': opt_beta2,
     }
    for opt in ['msgd_corr', 'adam']
    for opt_diagonal in ([0,1] if opt == 'msgd_corr' else [1])
    for opt_momentum, opt_beta2 in zip((1 - 10**rng.uniform(-1, -5, nrun)),
                                       (1 - 10**rng.uniform(-3, -5, nrun)))
  ])
  return all_hps


if __name__ == "__main__":
  # See atari_pol_eval.py
  args = parser.parse_args()
  if args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
    else:
      hps = all_hps[args.run]
      for k,v in hps.items():
        setattr(args, k, v)
      main(args)
  else:
    main(args)
