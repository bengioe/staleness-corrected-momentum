import gc

import torch
import numpy as np
from collections import namedtuple

try:
  from neural_network import tf, tint, get_device, set_device
except ImportError:
  from .neural_network import tf, tint, get_device, set_device


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



class ReplayBuffer:

  def __init__(self, seed, size, iwidth=84, near_strategy="both", extras=[]):
    self.rng = np.random.RandomState(seed)
    self.size = size
    self.near_strategy = near_strategy
    self.device = device = get_device()
    # Storing s,a,r,done,episode parity
    self.s = torch.zeros([size, iwidth, iwidth],
                         dtype=torch.uint8,
                         device=device)
    self.a = torch.zeros([size], dtype=torch.uint8, device=device)
    self.r = torch.zeros([size], dtype=torch.float32, device=device)
    self.t = torch.zeros([size], dtype=torch.uint8, device=device)
    self.p = torch.zeros([size], dtype=torch.uint8, device=device)
    self.idx = 0
    self.last_idx = 0
    self.maxidx = 0
    self.is_filling = True
    self._sumtree = SumTree(self.rng, size)
    self._sumtree = SumTree(self.rng, size)

  def compute_returns(self, gamma):
    if not hasattr(self, "g"):
      self.g = torch.zeros([self.size], dtype=torch.float32, device=self.device)
    g = 0
    for i in range(self.maxidx - 1, -1, -1):
      self.g[i] = g = self.r[i] + gamma * g
      if self.t[i]:
        g = 0

  def compute_reward_distances(self):
    if not hasattr(self, "rdist"):
      self.rdist = torch.zeros([self.size], dtype=torch.float32, device=self.device)

    d = 0
    for i in range(self.maxidx - 1, -1, -1):
      self.rdist[i] = d
      d += 1
      if self.r[i] != 0:
        d = 0


  def compute_values(self, fun, num_act, mbsize=128, nbins=256):
    if not hasattr(self, "last_v"):
      self.last_v = torch.zeros([self.size, num_act],
                                dtype=torch.float32,
                                device=self.device)
      self.vdiff_acc = np.zeros(nbins)
      self.vdiff_cnt = np.zeros(nbins)
      self.vdiff_bins = np.linspace(-1, 1, nbins-1)

    d = tint((-3, -2, -1, 0))  # 4 state history slice
    idx_0 = tint(np.arange(mbsize)) + 3
    idx_s = idx_0.reshape((-1, 1)) + d
    for i in range(int(np.ceil((self.maxidx - 4) / mbsize))):
      islice = idx_s + i * mbsize
      iar = idx_0 + i * mbsize
      if (i + 1) * mbsize >= self.maxidx - 2:
        islice = islice[:self.maxidx - i * mbsize - 2]
        iar = iar[:self.maxidx - i * mbsize - 2]
      s = self.s[islice].float().div_(255)
      with torch.no_grad():
        self.last_v[iar] = fun(s)
      if not i % 100:
        gc.collect()


  def compute_value_difference(self, sample, value):
    idxs = sample[-1]
    last_vs = self.last_v[idxs]
    diff = (last_vs - value).data.cpu().numpy().flatten()
    bins = np.digitize(diff, self.vdiff_bins)
    self.vdiff_acc += np.bincount(bins, diff, minlength=self.vdiff_acc.shape[0])
    self.vdiff_cnt += np.bincount(bins, minlength=self.vdiff_acc.shape[0])

  def update_values(self, sample, value):
    self.last_v[sample[-1]] = value.data


  def compute_episode_boundaries(self):
    self.episodes = []
    i = 0
    while self._sumtree.getp(i) == 0:
      i += 1
    print('start of first episode sampleable state', i)
    start = i
    while i < self.maxidx:
      if self.t[i]:
        self.episodes.append((start, i + 1))
        i += 1
        while i < self.maxidx and self._sumtree.getp(i) == 0:
          i += 1
        start = i
      i += 1
#    if i - start > 0:
#      self.episodes.append((start, i - 1))

  def compute_lambda_returns(self, fun, Lambda, gamma):
    if not hasattr(self, 'LR'):
      self.LR = LambdaReturn(Lambda, gamma)
      self.LG = torch.zeros([self.size], dtype=torch.float32, device=self.device)
    i = 0
    for start, end in self.episodes:
      s = self._idx2xy(tint(np.arange(start + 1, end)))[0]
      with torch.no_grad():
        vp = fun(s)[:, 0].detach()
        vp = torch.cat([vp, torch.zeros((1,), device=self.device)])
      self.LG[start:end] = self.LR(
          self.r[start:end],
          vp)
      i += 1

  def add(self, s, a, r, t, p, sampleable=1):
    self.s[self.idx] = torch.tensor(s[-1], dtype=torch.uint8).to(self.device)
    self.a[self.idx] = a
    self.r[self.idx] = r
    self.t[self.idx] = t * 1
    self.p[self.idx] = p
    self._sumtree.set(self.idx, sampleable)
    self.last_idx = self.idx
    self.idx += 1
    if self.idx >= self.size:
      self.idx = 0
      self.is_filling = False
    if self.is_filling:
      self.maxidx += 1
    if t:
      self.add(s,0,0,0,p,0) # pad end of episode with 1 unsampleable state

  def new_episode(self, s, p):
    # pad beginning of episode with 3 unsampleable states
    for i in range(3):
      self.add(s, 0, 0, 0, p, 0)

  def _idx2xy(self, idx, sidx=None):
    if sidx is None:
      d = tint((-3, -2, -1, 0, 1))  # 4 state history slice + 1 for s'
      sidx = (idx.reshape((idx.shape[0], 1)) + d) % self.maxidx
    return (
        self.s[sidx[:, :4]].float() / 255,
        self.a[idx],
        self.r[idx],
        self.s[sidx[:, 1:]].float() / 255,
        self.t[idx],
        idx,
    )

  def sample(self, n):
    idx = self._sumtree.stratified_sample(n)
    return self._idx2xy(idx)

  def slice_near(self, idx, dist=10, exclude_0=True):
    ar = np.arange(-dist, dist + 1)
    if exclude_0:
      ar = ar[ar != 0]
    sidx = (idx.reshape((-1, 1)) + tint(ar))
    p = self.p[idx]
    ps = self.p[sidx]
    pmask = (p[:, None] == ps).reshape((-1,)).float()
    sidx = sidx.reshape((-1,)) # clamp??
    return self._idx2xy(sidx), pmask

  def get(self, idx):
    return self._idx2xy(idx)

  def in_order_iterate(self, mbsize, until=None):
    if until is None:
      until = self.size
    valid_indices = np.arange(self.size)[self._sumtree.levels[-1] > 0]
    it = 0
    end = 0
    while end < valid_indices.shape[0]:
      end = min(it + mbsize, valid_indices.shape[0])
      if end > until:
        break
      yield self.get(tint(valid_indices[it:end]))
      it += mbsize


class SumTree:

  def __init__(self, rng, size):
    self.rng = rng
    self.nlevels = int(np.ceil(np.log(size) / np.log(2))) + 1
    self.size = size
    self.levels = []
    for i in range(self.nlevels):
      self.levels.append(np.zeros(min(2**i, size), dtype="float32"))

  def sample(self, q=None):
    q = self.rng.random() if q is None else q
    q *= self.levels[0][0]
    s = 0
    for i in range(1, self.nlevels):
      s *= 2
      if self.levels[i][s] < q and self.levels[i][s + 1] > 0:
        q -= self.levels[i][s]
        s += 1
    return s

  def stratified_sample(self, n):
    # As per Schaul et al. (2015)
    return tint([
        self.sample((i + q) / n)
        for i, q in enumerate(self.rng.uniform(0, 1, n))
    ])

  def set(self, idx, p):
    delta = p - self.levels[-1][idx]
    for i in range(self.nlevels - 1, -1, -1):
      self.levels[i][idx] += delta
      idx //= 2

  def getp(self, idx):
    return self.levels[-1][idx]


class PrioritizedExperienceReplay(ReplayBuffer):

  def __init__(self, *a, **kw):
    super().__init__(*a, **kw)
    self.sumtree = SumTree(self.rng, self.size)

  def sample(self, n, near=None, near_dist=5):
    if near is None:
      #try:
      idx = self.sumtree.stratified_sample(n).clamp(4, self.maxidx - 2)
    #except Exception as e:
    #    import pdb
    #    pdb.set_trace()
    else:
      raise ValueError("`near` argument incompatible with this class")
    return self._idx2xy(*self._fix_idx(idx))

  def set_last_priority(self, p):
    """sets the unnormalized priority of the last added example"""
    self.sumtree.set(self.last_idx, p)

  def set_prioties_at(self, ps, idx):
    self.sumtree.set(idx, ps)
    #for i, p in zip(idx, ps):
    #    self.sumtree.set(i, p)


if __name__ == "__main__":

  N = 1000000
  s = SumTree(np.random, 100)
  for i in range(100):
    s.set(i, np.random.random())
  s.sample(1)
  s.sample(1.2)
  x = [s.sample() for i in range(N)]
  u = np.unique(x, return_counts=True)[1] / N
  true_u = s.levels[-1] / np.sum(s.levels[-1])
  print(np.max(abs(u - true_u)))
  assert np.max(abs(u - true_u)) < 1e-3
