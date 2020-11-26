import gc

import torch
import numpy as np
from collections import namedtuple

from neural_network import tf, tint, get_device, set_device


class Minibatch(namedtuple('Minibatch', ['s', 'a', 'r', 'sp', 't', 'g', 'ap', 'idx', 'eidx'])):
  def slice(self, start, stop):
    return Minibatch(self.s[start:stop],
                     self.a[start:stop],
                     self.r[start:stop],
                     self.sp[start:stop],
                     self.t[start:stop],
                     self.g[start:stop],
                     self.ap[start:stop],
                     self.idx[start:stop],
                     self.eidx[start:stop])


class Episode:
  def __init__(self, gamma, tslice=4, snorm=255):
    self.s = []
    self.a = []
    self.r = []
    self.t = []
    self.ram = []
    self.gamma = gamma
    self.is_over = False
    self.device = get_device()
    self.tslice = tslice
    self.tslice_range = tint(np.arange(tslice)-tslice+1)
    self.snorm = snorm

  def add(self,s,a,r,t,ram_info=None):
    self.s.append(s)
    self.a.append(a)
    self.r.append(r)
    self.t.append(t)
    if ram_info is not None:
      self.ram.append(ram_info)
    if len(self.s) == 1: # is this the first frame?
      # Add padding frames
      for i in range(self.tslice-1):
        self.add(s,a,r,t,ram_info)

  def end(self):
    self.length = len(self.s)
    self.add(self.s[-1], self.a[-1], 0, 1,
             self.ram[-1] if len(self.ram) else None)
    self.s = torch.stack(self.s)
    self.a = torch.tensor(self.a, dtype=torch.uint8, device=self.device)
    self.r = torch.tensor(self.r, dtype=torch.float32, device=self.device)
    self.t = torch.tensor(self.t, dtype=torch.uint8, device=self.device)
    self.is_over = True
    self.ridx = []
    self.noridx = []
    for i in range(self.length):
      if self.r[i].item() != 0:
        self.ridx.append(i)
      else:
        self.noridx.append(i)
    self.ridx = np.int32(self.ridx)
    self.noridx = np.int32(self.noridx)
    gc.collect()

  def greturn(self, r):
    return (r * (self.gamma ** torch.arange(r.shape[0], device=r.device).float())).sum(0, True)

  def i2y(self, idx, n_step=1):
    if self.tslice > 1:
      sidx = (idx.reshape((idx.shape[0], 1)) + self.tslice_range)
    else:
      sidx = idx
    sidxp = torch.min(tint(self.length), sidx+n_step)
    idxp = torch.min(tint(self.length), idx+n_step)
    return (
        self.s[sidx].float() / self.snorm,
        self.a[idx],
        self.r[idx] if n_step == 1 else self.greturn(self.r[idx:idxp]),
        self.s[sidxp].float() / self.snorm,
        self.t[idxp-1],
        self.g[idx],
        self.a[idxp],
        idx,
    )

  def compute_returns(self):
    g = []
    G = 0
    for t in range(self.length, -1, -1):
      G = self.gamma * G + self.r[t].item()
      g.append(G)
    self.g = torch.tensor(g[::-1], dtype=torch.float32, device=self.device)

  def sample(self, n_step=1):
    if 1:
      return self.i2y(tint([np.random.randint(self.tslice-1,self.length-1)]), n_step=n_step)
    r = np.random.randint(0, 2)
    if r == 0:
      return self.i2y(tint([np.random.choice(self.ridx)]))
    return self.i2y(tint([np.random.choice(self.noridx)]))

class ReplayBufferV2:
  def __init__(self, seed, size, gamma=0.99, tslice=4, snorm=255, xdtype=torch.uint8):
    self.current_size = 0
    self.size = size
    self.device = get_device()
    self.rng = np.random.RandomState(seed)
    self.gamma = gamma
    self.hit_max = False
    self.current_episode = Episode(self.gamma, tslice=tslice, snorm=snorm)
    self.episodes = []
    self.tslice = tslice
    self.snorm = snorm
    self.xdtype = xdtype

  def add(self, s,a,r,t, ram_info=None):
    s = torch.tensor(s[-1] if self.tslice > 1 else s, dtype=self.xdtype).to(self.device)
    self.current_episode.add(
        s,
        a,
        r,
        t * 1,
        ram_info=ram_info)
    if t:
      self.current_episode.end()
      self.current_episode.compute_returns()
      self.current_size += self.current_episode.length
      self.episodes.append(self.current_episode)
      self.current_episode = Episode(self.gamma, tslice=self.tslice, snorm=self.snorm)
      while self.current_size > self.size:
        self.hit_max = True
        e = self.episodes.pop(0)
        self.current_size -= e.length

  def sample(self, n, n_step=1):
    eidx = self.rng.randint(0, len(self.episodes), n)
    data = [self.episodes[i].sample(n_step=n_step) for i in eidx]
    return Minibatch(*[torch.cat([d[i] for d in data])
                       for i in range(len(data[0]))], eidx)

  def iterate(self, mbsize):
    eidx = 0
    t = 0
    z = torch.zeros(mbsize, device=self.device)
    while True:
      mb = []
      while len(mb) < mbsize:
        mb.append(self.episodes[eidx].i2y(tint([t])))
        t += 1
        if t >= self.episodes[eidx].length:
          t = 0
          eidx += 1
          if eidx >= len(self.episodes):
            break
      if len(mb):
        yield Minibatch(*[torch.cat([d[i] for d in mb])
                          for i in range(len(mb[0]))], z[:len(mb)])
      if eidx >= len(self.episodes) or len(mb) < mbsize:
        break
