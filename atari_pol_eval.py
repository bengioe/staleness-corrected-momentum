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

from scmsgd import SCMSGD, OptChain

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--nhid", default=32, help="#hidden", type=int)
parser.add_argument("--buffer_size", default=50000, help="Replay buffer size",type=int)
parser.add_argument("--num_iterations", default=100000, type=int)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--head_type", default='normal', help='normal or slim')
parser.add_argument("--body_type", default='normal', help='normal or slim')
parser.add_argument("--opt", default='msgd')
parser.add_argument("--opt_momentum", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.999, type=float)
parser.add_argument("--opt_diagonal", default=True, type=int)
parser.add_argument("--opt_ignore_vprime", default=0, type=int)
parser.add_argument("--opt_correct_adam", default=0, type=int)
parser.add_argument("--num_full_corr", default=0, type=int,
                    help="End parameters for which the full correction matrix (instead of diagonal) is computed")
parser.add_argument("--target_type", default='none') # frozen, none, moving
parser.add_argument("--target_clone_interval", default=10000, type=int)
parser.add_argument("--target_tau", default=0.01, type=float)
parser.add_argument("--checkpoint_freq", default=10000, type=int)
parser.add_argument("--test_freq", default=2500, type=int)
parser.add_argument("--env_name", default='ms_pacman')
parser.add_argument("--device", default='cuda', help="device")
parser.add_argument("--progress", default=False, action='store_true', help='display a progress bar')
parser.add_argument("--array", default='')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--test_run", default=False, action='store_true')
parser.add_argument("--save_path", default='results/')
parser.add_argument("--save_parameters", default=False, type=bool)




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
    # This chains two optimizers, one with the early (lower layers)
    # parameters, and the second with the last num_full_corr
    # parameters (presumably the parameters close to the value
    # prediction).  The first one is diagonal, while the second
    # maintains the full matrix (which would be way too large for the
    # entire model). To make this tractable, it is recommended to use
    # head_type == 'slim', which adds a slimmer head with fewer
    # parameters to maintain an O(n^2) matrix of.
    return OptChain([
      SCMSGD(sub_theta, args.learning_rate, weight_decay=args.weight_decay,
               momentum=args.opt_momentum,
               dampening=args.opt_momentum,
               diagonal=diag,
               correct_adam=args.opt_correct_adam,
               beta2=args.opt_beta2)
      for sub_theta, diag in ([(theta[:-args.num_full_corr], args.opt_diagonal),
                               (theta[-args.num_full_corr:], False)] # Force non-diagonal
                              if args.num_full_corr > 0 else
                              [(theta, args.opt_diagonal)])])
  elif args.opt == "rmsprop":
    return torch.optim.RMSprop(theta, args.learning_rate, weight_decay=args.weight_decay)
  elif args.opt == "adam":
    return torch.optim.Adam(theta, args.learning_rate, weight_decay=args.weight_decay,
                            betas=(args.opt_momentum, args.opt_beta2))
  else:
    raise ValueError(args.opt)


def load_expert(env_name, env):
  model_path = f"oracles/{env_name}.pth"
  with open(model_path, "rb") as f:
    m = torch.load(f)

  device = mm.get_device()
  dqn = DQN(
      odict({
          "history_length": 4,
          "hidden_size": 256,
          "architecture": "data-efficient",
          "atoms": 51,
          "noisy_std": 0.1,
          "V_min": -10,
          "V_max": 10,
          "device": device,
      }), env.num_actions)
  dqn.load_state_dict(m)
  dqn.eval()
  dqn.to(device)
  return dqn

def fill_buffer_with_expert(dqn, env, replay_buffer):
  totr = 0
  obs = env.reset()
  it = 0
  device = mm.get_device()
  while not replay_buffer.hit_max:
    action = dqn.act_e_greedy(
        torch.tensor(obs).float().to(device) / 255, epsilon=0.01)
    obsp, r, done, tr = env.step(action)
    replay_buffer.add(obs, action, r, done)
    obs = obsp
    totr += tr
    if done:
      print("Done episode %d reward %d"%(totr, replay_buffer.current_size))
      totr = 0
      obs = env.reset()
      if args.test_run:
        break
    it += 1
  return dqn

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
  #env.r_gamma = gamma
  checkpoint_freq = args.checkpoint_freq
  test_freq = args.test_freq
  target_tau = args.target_tau
  target_clone_interval = args.target_clone_interval
  target_type = args.target_type
  num_iterations = args.num_iterations
  num_Q_outputs = env.num_actions

  # Model
  act = torch.nn.LeakyReLU()
  if args.body_type == 'normal':
    body = torch.nn.Sequential(torch.nn.Conv2d(4, nhid, 8, stride=4, padding=4), act,
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
  do_set_predictions = args.opt == 'msgd_corr'

  # Replay Buffer
  replay_buffer = ReplayBufferV2(seed, args.buffer_size)
  test_replay_buffer = ReplayBufferV2(seed, 10000)

  # Get expert trajectories
  expert = load_expert(args.env_name, env)
  fill_buffer_with_expert(expert, env, replay_buffer)
  fill_buffer_with_expert(expert, env, test_replay_buffer)

  ar = lambda x: torch.arange(x.shape[0], device=x.device)

  losses = []
  num_iterations = 1 + num_iterations
  ignore_vprime = bool(args.opt_ignore_vprime)
  # Run policy evaluation
  for it in (tqdm(range(num_iterations), smoothing=0) if args.progress else range(num_iterations)):
    sample = replay_buffer.sample(mbsize)

    q = Qf(sample.s)
    v = q[ar(q), sample.a.long()]
    vp = Qf_target(sample.sp)[ar(q), sample.ap.long()] # Sarsa updat
    gvp = (1 - sample.t.float()) * gamma * vp
    loss = (v - (sample.r + gvp.detach())).pow(2)
    _loss = loss
    if do_set_predictions:
      opt.set_predictions(v.mean(), gvp.mean() if not ignore_vprime else None)

    loss = loss.mean()
    loss.backward(retain_graph=True)
    opt.step()
    opt.zero_grad()

    losses.append(loss.item())

    if target_type == 'frozen' and it % target_clone_interval == 0:
      Qf_target = copy.deepcopy(Qf)
    elif target_type == 'moving':
      for target_param, param in zip(Qf_target.parameters(), Qf.parameters()):
        target_param.data.mul_(1-target_tau).add_(param, alpha=target_tau)

    if it % checkpoint_freq == 0 and args.save_parameters:
      with params_conn.begin(write=True) as txn:
        txn.put(f'parameters_{it}'.encode(), packobj(Qf.state_dict()))

    if it % test_freq == 0:
      expert_q_loss = 0
      expert_v_loss = 0
      mc_loss = 0
      n = 0
      with torch.no_grad():
        for sample in test_replay_buffer.iterate(512):
          n += sample.s.shape[0]
          q = Qf(sample.s)[ar(sample.a), sample.a.long()]
          mc_loss += (q - sample.g).pow(2).sum().item()
      print(q.shape, sample.g.shape)
      with results_conn.begin(write=True) as txn:
        txn.put(f'expert-loss_{it}'.encode(), packobj((expert_q_loss/n,
                                                       expert_v_loss/n,
                                                       mc_loss/n)))
        if it > 0:
          txn.put(f'train-loss_{it}'.encode(), packobj(losses))
        print(it, np.mean(losses), (expert_q_loss/n, expert_v_loss/n, mc_loss/n))
        losses = []


    if np.isnan(loss.item()):
      print("Learning has diverged, nan loss")
      with results_conn.begin(write=True) as txn:
        txn.put(b'diverged', b'True')
      break
  print("Done.")


def array_aug_24(args):
  base = {'mbsize': 32, 'buffer_size': 100_000,
          'num_iterations': 500_000,
          'opt_beta2': 0.999,
          'nhid': 64,
          'target_type': 'none',}
  all_hps = ([
    {**base,
     'opt': 'msgd_corr',
     'opt_momentum': 0.9,
     'nhid': 64,
     'learning_rate': 1e-4,
     }
    for seed in range(20)
  ] + [
    {**base,
     'opt': 'adam',
     'opt_momentum': 0.99,
     'nhid': 64,
     'learning_rate': 5e-5,
     }
    for seed in range(20)
  ])

  return all_hps


if __name__ == "__main__":
  # To launch the array above on e.g. slurm, one would launch an array
  # of jobs from 0-N with the command
  #      python atari_pol_eval.py --array=array_aug_24 \
  #             --run=$SLURM_ARRAY_TASK_ID --save_path=results/aug_24/
  # N being len(all_hps) as below
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
