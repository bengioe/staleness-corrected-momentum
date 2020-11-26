import torch
import torch.nn as nn
import numpy as np
import lmdb
import tqdm
import gzip
import pickle
import os.path
import math
import copy
import time

import numpy as np
from backpack import extend

from replay_buffer import ReplayBufferV2, set_device
from scmsgd import SCMSGD, FixedSGD, OptChain, SCMTDProp
from classical_envs import MountainCarEnv, AcrobotEnv, CartpoleEnv, LunarLanderEnv
from rbf import RBFGrid

_progress = False

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


def init_weights_2(m):
  if isinstance(m, torch.nn.Linear):
    k = 2 / (np.sum(m.weight.shape))
    m.weight.data.normal_(0, k)
    m.bias.data.fill_(0)
  if isinstance(m, torch.nn.Conv2d):
    u,v,w,h = m.weight.shape
    k = (2 / (u + w*h*v))
    m.weight.data.normal_(0, k)
    m.bias.data.fill_(0)

def main(path, hps):

    k = str(hps['id']).encode()

    #dev = torch.device('cuda')
    dev = torch.device('cpu')
    set_device(dev)

    conn = lmdb.open(path, map_size=int(16 * 2 ** 30)) # 16gb max?
    if 0:
        with conn.begin() as txn:
            cursor = txn.cursor()
            if cursor.set_key(k + b'_lock'): # Some other process is running this
                return
    with conn.begin(write=True) as txn:
        txn.put(k + b'_lock', b'lock')

    seed = 142857 + hps['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_cls = {'mountain_car': MountainCarEnv,
               'acrobot': AcrobotEnv,
               'cartpole': CartpoleEnv,
               'lunarlander': LunarLanderEnv,
               }[hps.get('env', 'mountain_car')]

    env = env_cls(seed)
    act = {'lrelu': nn.LeakyReLU(),
           'elu': nn.ELU(),
           'tanh': nn.Tanh(),
    }[hps.get('act', 'lrelu')]
    gamma = {'mountain_car': 0.95,
             'acrobot': 0.99,
             'cartpole': 0.99,
             'lunarlander': 0.99,
            }[hps.get('env', 'mountain_car')]

    oracle = env_cls.get_oracle()
    oracle.to(dev)

    nhid = hps.get('nhid', 16)
    mbsize = hps['mbsize']
    learning_rate = hps['lr']
    beta = hps['beta']
    beta2 = hps.get('beta2', 0)
    dampening = hps['beta'] if hps.get('dampening', True) else 0
    mode = hps['mode']
    online_experiment = hps.get('online_experiment', False)
    num_steps = hps.get('num_steps', 5000)
    arch = hps.get('arch', 'mlp')
    nin = env._env.observation_space.shape[0]
    n_rbf_grid = hps.get('n_rbf_grid', 20)
    rbf_sigma = hps.get('rbf_sigma', 1.5)
    measure_drift = hps.get('measure_drift', False)
    frozen_target = hps.get('frozen_target', False)
    frozen_target_update_freq = hps.get('frozen_target_update_freq', 100)
    gen_buffer = hps.get('gen_buffer', False)
    buffer_size = hps.get('buffer_size', 10000)
    td_n_step = hps.get('td_n_step', 1)
    corr_layers = hps.get('corr_layers', None)
    block_size = hps.get('block_size', 0)
    disable_corr = hps.get('disable_corr', False)

    if td_n_step > 1 and not gen_buffer:
      raise NotImplementedError('td_n_step > 1 and not gen_buffer')

    if arch == 'mlp':
        model = nn.Sequential(nn.Linear(nin, nhid), act,
                              nn.Linear(nhid, nhid), act,
                              nn.Linear(nhid, nhid), act,
                              nn.Linear(nhid, 1))
        model.to(dev)
        if frozen_target:
            target_model = copy.deepcopy(model)

    if arch == 'mlp_tall':
        model = nn.Sequential(nn.Linear(nin, nhid), act,
                              nn.Linear(nhid, nhid), act,
                              nn.Linear(nhid, nhid), act,
                              nn.Linear(nhid, nhid), act,
                              nn.Linear(nhid, nhid), act,
                              nn.Linear(nhid, 1))
        model.to(dev)

    elif arch == 'mlp_small':
        model = nn.Sequential(nn.Linear(nin, nhid), act,
                              nn.Linear(nhid, 1))
        model.to(dev)
        if frozen_target:
            target_model = copy.deepcopy(model)

    elif arch == 'linear_rbf':
        model = nn.Sequential(RBFGrid(nin, n_rbf_grid, sigma=rbf_sigma, device=dev),
                              nn.Linear(n_rbf_grid**nin, 1))
        model.to(dev)

    model = extend(model)

    paramslices = [0] + list(np.cumsum([np.prod(i.shape) for i in model.parameters()]))
    p2v = lambda x: torch.cat([i.reshape(-1) if i is not None else torch.zeros_like(p).reshape(-1)
                               for i, p in zip(x, model.parameters())])
    v2p = lambda x: [x[u:v].reshape(p.shape)
                     for u,v,p in zip(paramslices[:-1], paramslices[1:], model.parameters())]

    opt_normal = FixedSGD(model.parameters(),
                          learning_rate, momentum=beta, dampening=dampening)
    rng = np.random.RandomState(seed)
    past_data = []
    horizon = int(np.ceil(2 * 1 / (1-beta)))
    losses = []
    true_losses = []
    mc_losses = []
    weighted_vt_drift = []
    weighted_vtp_drift = []

    if gen_buffer:
        replay_buffer = ReplayBufferV2(seed, buffer_size + 2000, gamma, 1, 1, torch.float32)
        s = env.reset()
        for i in (tqdm.tqdm if _progress else lambda x:x)(range(buffer_size)):
            a = env.optimal_action()
            sp, r, t, _ = env.step(a)
            replay_buffer.add(s, a, r, t)
            s = env.reset() if t else sp

    if 'corr' in mode or 'tdprop' in mode:
      if corr_layers is None:
        if 1:
          opt_corr = SCMTDProp(model.parameters(), learning_rate, momentum=beta,
                               diagonal='diag' in mode, dampening=dampening,
                               beta2=beta2, block_diagonal=block_size,
                               disable_corr=disable_corr)
          opt_corr.mom_correct_bias = False
        else:
          opt_corr = SCMSGD(model.parameters(), learning_rate, momentum=beta,
                            diagonal='diag' in mode, dampening=dampening,
                            beta2=beta2)
      else:
        raise ValueError('fixme')
        params = list(model.parameters())
        layers = [params[i*2:i*2+2] for i in range(len(params)//2)]
        to_corr = sum([layers[i] for i in corr_layers], [])
        others = sum([layers[i] for i in range(len(layers)) if i not in corr_layers], [])
        if 0: # correct with SGD or diagonal SCMSGD?
          opt_corr = OptChain([
            SCMSGD(to_corr, learning_rate, momentum=beta,
                     diagonal='diag' in mode, dampening=dampening, beta2=beta2)] + ([
                       FixedSGD(others, learning_rate, momentum=beta, dampening=dampening),
                     ] if others else []))
        else:
          opt_corr = OptChain(([
            SCMSGD(to_corr, learning_rate, momentum=beta,
                     diagonal=False, dampening=dampening, beta2=beta2)]) +  ([
                       SCMSGD(others, learning_rate, momentum=beta,
                                diagonal=True, dampening=dampening, beta2=beta2),
                     ] if others else []))

    if online_experiment:
        o_st = env.reset()
        assert mbsize == 1

    for i in (tqdm.tqdm if _progress else lambda x:x)(range(num_steps)):
        if not gen_buffer:
            if online_experiment:
                at = env.optimal_action()
                stp, r, done,_ = env.step(at)
                transitions = [(o_st, stp, r, done)]
                o_st = env.reset() if done else stp
            else:
                transitions = [(env.reset(), *env.step(env.optimal_action()))
                               for i in range(mbsize)]
            st = torch.tensor([i[0] for i in transitions]).float().to(dev)
            stp = torch.tensor([i[1] for i in transitions]).float().to(dev)
            r = torch.tensor([i[2] for i in transitions]).float().to(dev)
            done = torch.tensor([i[3] for i in transitions]).float().to(dev)
        else:
            mb = replay_buffer.sample(mbsize, n_step=td_n_step)
            st, stp, r, done = mb.s, mb.sp, mb.r, mb.t

        #vt = model(st).squeeze(1)
        #vtp = model(stp).squeeze(1) if not frozen_target else target_model(stp).squeeze(1)
        if not frozen_target:
          o = model(torch.cat([st, stp], 0)).squeeze(1)
          vt = o[:mbsize]
          vtp = o[mbsize:]
        else:
          vt = model(st).squeeze(1)
          vtp = target_model(stp).squeeze(1)
        gamma_mask = (gamma**td_n_step) * (1-done)
        target = (r + vtp * gamma_mask)
        loss = (vt - target.detach()).pow(2)

        with torch.no_grad():
            voracle = oracle(st).squeeze(1)
            true_loss = (voracle - vt).pow(2).mean()
            if gen_buffer:
                mc_losses.append((vt-mb.g).pow(2).mean().item())
        if not i % 100 and _progress:
          if gen_buffer:
            print('tl, mc, l', true_loss.item(), mc_losses[-1], loss.mean().item())
            print('oracle - g',(voracle - mb.g).pow(2).mean().item())
            print(done.float().mean())
          else:
            print(true_loss.item(), loss.mean().item(), done.mean())
        trans = [(st, stp, r, done, vt.detach().clone(), vtp.detach().clone())]

        true_losses.append(true_loss.item())
        losses.append(loss.mean().item())
        if mode == 'normal' or mode == 'largebatch':
            loss.mean().backward()
            opt_normal.step()
            opt_normal.zero_grad()
        if mode == 'gtd':
            loss = (vt - target).pow(2)
            loss.mean().backward()
            opt_normal.step()
            opt_normal.zero_grad()
        if mode == 'oracle':
            (loss*(1-dampening)).mean().backward()

            if len(past_data):
                st = torch.cat([i[0] for i in past_data], 0)
                stp = torch.cat([i[1] for i in past_data], 0)
                r = torch.cat([i[2] for i in past_data], 0)
                done = torch.cat([i[3] for i in past_data], 0)
                b = torch.cat([torch.ones(mbsize) * beta ** (len(past_data) - t) * (1-dampening)
                               for t in range(len(past_data))], 0).to(dev)
                vt = model(st).squeeze(1)
                vtp = model(stp).squeeze(1).detach()
                li = (vt - (r + (gamma**td_n_step) * vtp * (1-done))).pow(2).mul(b).sum() / mbsize
                li.backward()
            for p in model.parameters():
                p.data -= p.grad * learning_rate
                p.grad.fill_(0)
        if 'corr' in mode or 'tdprop' in mode:
          if 0:
            opt_corr.set_predictions(vt.mean(),
                                     None if 'novp' in mode else target.mean())
            loss.mean().backward(retain_graph=True)
            opt_corr.step()
            opt_corr.zero_grad()
          else:
            opt_corr.backward_and_step(vt, vtp, vt - target, gamma_mask)
        past_data = past_data[-horizon+1:] + trans

        if measure_drift:
            st = torch.cat([i[0] for i in past_data], 0)
            stp = torch.cat([i[1] for i in past_data], 0)
            r = torch.cat([i[2] for i in past_data], 0)
            done = torch.cat([i[3] for i in past_data], 0)
            vt0 = torch.cat([i[4] for i in past_data], 0)
            vtp0 = torch.cat([i[5] for i in past_data], 0)
            b = torch.cat([torch.ones(mbsize) * beta ** (len(past_data) - t) * (1-dampening)
                           for t in range(len(past_data))], 0).to(dev)
            with torch.no_grad():
                vt = model(st).squeeze(1)
                vtp = model(stp).squeeze(1).detach()
            weighted_vt_drift.append(((vt0 - vt).pow(2).mul(b).sum() / mbsize).item())
            weighted_vtp_drift.append((((vtp0 - vtp) * (1-done)).pow(2).mul(b).sum() / mbsize).item())
        if frozen_target and i and not i % frozen_target_update_freq:
            target_model = copy.deepcopy(model)
    losses = np.float32(losses)
    true_losses = np.float32(true_losses)
    weighted_vt_drift = np.float32(weighted_vt_drift)
    weighted_vtp_drift = np.float32(weighted_vtp_drift)


    with conn.begin(write=True) as txn:
        k = str(hps['id']).encode()
        txn.put(k + b'_losses', gzip.compress(pickle.dumps(losses)))
        txn.put(k + b'_true_losses', gzip.compress(pickle.dumps(true_losses)))
        txn.put(k + b'_mc_losses', gzip.compress(pickle.dumps(mc_losses)))
        txn.put(k + b'_model', gzip.compress(pickle.dumps(model.state_dict())))
        txn.put(k + b'_hps', gzip.compress(pickle.dumps(hps)))
        txn.put(k + b'_wdrifts', gzip.compress(pickle.dumps((weighted_vt_drift, weighted_vtp_drift))))
        txn.pop(k + b'_lock')



if __name__ == '__main__':

    # Figure 3
    if 0:
      path = 'results/td_pe_mcar_replay'
      all_hps = [{'nhid': nhid,
                  'beta': beta,
                  'mbsize': mbsize,
                  'num_steps': 5_000,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(10)
                 for nhid in [16, 32]
                 for beta in [0.9, 0.99]
                 for lr in [0.5, 0.1, 0.05]
                 for mode in ['normal', 'oracle', 'corr', 'corr_diag']
                 for mbsize in [4,16,64]]
    # Figure 4
    if 0:
      path = 'results/td_pe_mcar_online'
      all_hps = [{'nhid': nhid,
                  'beta': beta,
                  'mbsize': 1,
                  'online_experiment': True,
                  'num_steps': 20_000,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for nhid in [16]
                 for beta in [0.9, 0.99]
                 for lr in [0.0005, 0.001, 0.005]
                 for mode in ['normal', 'oracle', 'corr', 'corr_diag']
                 for seed in range(10)]

    # Mbsize vs loss
    if 0:
        path = 'results/td_pe_mcar_mbsize_vs_loss'
        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'num_steps': 5_000,
                    'lr': lr,
                    'seed': seed}
                   for seed in range(20)
                   for nhid in [16]
                   for beta in [0.99]
                   for mbsize in [1,4,16,64]
                   for lr in [5e-3, 1e-2, 1e-1]
                   for mode in ['corr', 'corr_diag', 'normal', 'oracle']]

        path = 'results/td_pe_mcar_mbsize_vs_loss_2'
        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'num_steps': 5_000,
                    'lr': lr,
                    'seed': seed}
                   for seed in range(5)
                   for nhid in [16]
                   for beta in [0.99]
                   for mbsize in [1,4,16,64]
                   for lr in [1e-1]
                   for mode in ['corr', 'corr_diag', 'normal', 'oracle']]

        path = 'results/td_pe_mcar_mbsize_vs_loss_3'
        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'num_steps': 5_000,
                    'lr': lr,
                    'seed': seed}
                   for seed in range(5)
                   for nhid in [16]
                   for beta in [0.99]
                   for mbsize in [1,4,16,64]
                   for lr in [1e-1]
                   for mode in ['corr2', 'corr2_diag']]

        path = 'results/td_pe_mcar_mbsize_vs_loss_4'
        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'num_steps': 5_000,
                    'lr': lr,
                    'seed': seed}
                   for seed in range(5)
                   for nhid in [16]
                   for beta in [0.99]
                   for mbsize in [1,4,16,64]
                   for lr in [1e-1]
                   for mode in ['corr3', 'corr3_diag']]

        path = 'results/td_pe_mcar_mbsize_vs_loss_5'
        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'num_steps': 5_000,
                    'lr': lr,
                    'seed': seed}
                   for seed in range(5)
                   for nhid in [16]
                   for beta in [0.99]
                   for mbsize in [1,4,16,64]
                   for lr in [1e-1]
                   for mode in ['corr_fix', 'corr_fix_diag']]

        path = 'results/td_pe_mcar_mbsize_vs_loss_6'
        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'num_steps': 5_000,
                    'lr': lr,
                    'seed': seed}
                   for seed in range(5)
                   for nhid in [16]
                   for beta in [0.99]
                   for mbsize in [1,4,16,64]
                   for lr in [1e-1]
                   for mode in ['corr_true', 'corr_true_diag']]

    # Beta vs loss
    if 0:
        path = 'results/td_pe_mcar_beta_vs_loss'
        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'num_steps': 5_000,
                    'lr': lr,
                    'seed': seed}
                   for seed in range(4)
                   for nhid in [16]
                   for beta in [0.8, 0.9, 0.95,0.975, 0.99, 0.995, 0.999]
                   for mbsize in [4]
                   for lr in [1e-2, 5e-2]
                   for mode in ['corr', 'corr_diag', 'normal', 'oracle']]

    # Cartpole
    if 0:
        path = 'results/td_pe_cartpole'
        all_hps = [{
            'env': 'cartpole',
            'gen_buffer': True,
            'buffer_size': 50_000,
            'nhid': nhid,
            'beta': beta,
            'mbsize': mbsize,
            'num_steps': 5_000,
            'td_n_step': 5,
            'mode': mode,
            'lr': lr,
            'seed': seed}
           for seed in range(10)
           for nhid in [8, 16, 32]
           for beta in [0.9, 0.95, 0.99]
           for lr in [5e-2, 5e-1, 1e-2]
           for mode in ['normal', 'oracle', 'corr', 'corr_diag']
           for mbsize in [8, 16, 32]]

    # Acrobot
    if 0:
        path = 'results/td_pe_acrobot'
        all_hps = [{
            'env': 'acrobot',
            'gen_buffer': True,
            'buffer_size': 50_000,
            'nhid': nhid,
            'beta': beta,
            'mbsize': mbsize,
            'num_steps': 5_000,
            'td_n_step': nstep,
            'mode': mode,
            'lr': lr,
            'seed': seed}
           for seed in range(10)
           for nhid in [8, 16, 32]
           for beta in [0.9, 0.95, 0.99]
           for lr in [1e-1, 5e-2, 1e-2]
           for nstep in [2, 3]
           for mode in ['normal', 'oracle', 'corr', 'corr_diag']
           for mbsize in [8, 16, 32]]


    # Frozen Target
    if 0:
      path = 'results/td_pe_mcar_replay_frozen'
      all_hps = [{'nhid': nhid,
                  'beta': beta,
                  'mbsize': mbsize,
                  'num_steps': 5_000,
                  'frozen_target': True,
                  'frozen_target_update_freq': 100,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(10)
                 for nhid in [16, 32]
                 for beta in [0.9, 0.99]
                 for lr in [0.5, 0.1, 0.05]
                 for mode in ['normal']
                 for mbsize in [4,16,64]]

    # Figure 3 long
    if 0:
      path = 'results/td_pe_mcar_replay_longer'
      all_hps = [{'nhid': nhid,
                  'beta': beta,
                  'mbsize': mbsize,
                  'num_steps': 50_000,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(10)
                 for nhid in [32]
                 for beta in [0.9]
                 for lr in [0.5, 0.1, 0.05]
                 for mode in ['normal', 'oracle', 'corr', 'corr_diag']
                 for mbsize in [16]]
    # Figure 3 more lr
    if 0:
      path = 'results/td_pe_mcar_replay_stepsize'
      all_hps = [{'nhid': nhid,
                  'beta': beta,
                  'mbsize': mbsize,
                  'num_steps': 5_000,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(10)
                 for nhid in [32]
                 for beta in [0.9]
                 for lr in [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001]
                 for mode in ['normal', 'oracle', 'corr', 'corr_diag']
                 for mbsize in [4]]

    if 0:
      path = 'results/td_pe_mcar_replay_tdprop_2'
      all_hps = [{'nhid': nhid,
                  'beta': beta if 'corr' in mode else 0,
                  'beta2': 0.999 if 'tdprop' in mode else 0,
                  'mbsize': mbsize,
                  'num_steps': 5_000,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(5)
                 for nhid in [32]
                 for beta in [0.9]
                 for mode in ['corr_tdprop_diag', 'corr', 'corr_diag', 'tdprop', 'corr_tdprop']
                 for lr in (np.logspace(-4, -2.5, 10) if 'tdprop' in mode else
                            np.logspace(-2.5, -0.5, 10))
                 for mbsize in [16]]

    if 1:
      path = 'results/td_pe_mcar_replay_blockd'
      all_hps = [{'nhid': nhid,
                  'beta': beta if 'corr' in mode else 0,
                  'beta2': 0.999 if 'tdprop' in mode else 0,
                  'mbsize': mbsize,
                  'num_steps': 5_000,
                  'block_size': block_size,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(5)
                 for nhid in [32]
                 for beta in [0.9]
                 for block_size in [2,4,8,16,32]
                 for mode in ['corr_tdprop_blockd', 'corr_blockd']
                 for lr in ([1.5e-3] if 'tdprop' in mode else
                            [2e-1])
                 for mbsize in [16]]

      path = 'results/td_pe_mcar_replay_blockd_2'
      all_hps = [{'nhid': nhid,
                  'beta': beta if 'corr' in mode else 0,
                  'beta2': 0.999 if 'tdprop' in mode else 0,
                  'mbsize': mbsize,
                  'num_steps': 5_000,
                  'block_size': block_size,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(5)
                 for nhid in [32]
                 for beta in [0.9]
                 for block_size in [0]
                 for mode in ['corr_tdprop_diag', 'corr_diag', 'corr', 'corr_tdprop']
                 for lr in ([1.5e-3] if 'tdprop' in mode else
                            [2e-1])
                 for mbsize in [16]]

      path = 'results/td_pe_mcar_replay_blockd_3'
      all_hps = [{'nhid': nhid,
                  'beta': beta if 'mom' in mode else 0,
                  'beta2': 0.999 if 'tdprop' in mode else 0,
                  'disable_corr': True,
                  'mbsize': mbsize,
                  'num_steps': 5_000,
                  'mode': mode,
                  'lr': lr,
                  'seed': seed}
                 for seed in range(5)
                 for nhid in [32]
                 for beta in [0.9]
                 for block_size in [0]
                 for mode in ['tdprop', 'tdprop_mom']
                 for lr in ([1.5e-3, 2e-4, 1e-4, 1e-3, 5e-3])
                 for mbsize in [16]]



    for i, u in enumerate(all_hps):
        u['id'] = i
    print(len(all_hps), 'experiments')


    if 0:
        _progress = True
        import importlib
        import scmsgd
        importlib.reload(scmsgd)
        from scmsgd import SCMSGD, FixedSGD, OptChain, SCMTDProp
        main(path, all_hps[0])
    else:
        torch.set_num_threads(1)
        import ray
        ray.init(num_cpus=8)

        conn = lmdb.open(path, map_size=int(16 * 2 ** 30)) # 16gb max?

        rmain = ray.remote(main)
        with conn.begin() as txn:
            cursor = txn.cursor()
            jobs = [rmain.remote(path, i) for i in all_hps
                    if (not cursor.set_key(str(i['id']).encode() + b'_hps'))]
        print(len(jobs), 'jobs')

        with tqdm.tqdm(total=len(jobs), smoothing=0) as t:
            while len(jobs):
                ready, jobs = ray.wait(jobs)
                for i in ready:
                    t.update()
