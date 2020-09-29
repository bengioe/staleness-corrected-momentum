import torch
import torch.nn as nn
import numpy as np
import lmdb
import tqdm
import gzip
import pickle


def main(path, hps):
    # Make easy regression data
    N = 10000
    rng = np.random.RandomState(142857)
    x = np.float32(rng.uniform(-1, 1, (N,)))
    yf = lambda x:(
           0.38 * np.sin(x * 12)
         + 0.5 * np.sin(2.14 * (x + 2))# * rng.normal(0,0.05,(x.shape[0],)))
         + 0.82 * np.sin(9 * x + 0.4)
         + 0.322 * np.sin(38 * x - 0.102)
    )
    y = yf(x)


    act = nn.LeakyReLU()
    nhid = hps['nhid']
    mbsize = hps['mbsize']
    learning_rate = hps['lr']
    beta = hps['beta']
    mode = hps['mode']
    seed = 142857 + hps['seed'] + 12938

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nn.Sequential(nn.Linear(1, nhid), act,
                          nn.Linear(nhid, nhid), act,
                          nn.Linear(nhid, nhid), act,
                          nn.Linear(nhid, 1))
    opt_normal = torch.optim.SGD(model.parameters(),
                                 learning_rate, momentum=beta)
    paramslices = [0] + list(np.cumsum([np.prod(i.shape) for i in model.parameters()]))
    #print(paramslices[-1],'parameters')
    p2v = lambda x: torch.cat([i.reshape(-1) if i is not None else torch.zeros_like(p).reshape(-1)
                               for i, p in zip(x, model.parameters())])
    v2p = lambda x: [x[u:v].reshape(p.shape)
                     for u,v,p in zip(paramslices[:-1], paramslices[1:], model.parameters())]
    if 'corr' in mode:
        mu_bar = torch.zeros(paramslices[-1])
        Z = torch.zeros((paramslices[-1], paramslices[-1]))
        q = torch.zeros(paramslices[-1])


    rng = np.random.RandomState(seed)
    past_data = []
    horizon = int(np.ceil(2 / (1-beta)))
    losses = []
    dampen = 1 # 1-beta
    for i in range(5000):
        idx = rng.randint(0, N, (mbsize,))
        xi = torch.tensor(x[idx]).unsqueeze(1)
        yi = torch.tensor(y[idx])
        o = model(xi).squeeze(1)
        loss = (o - yi).pow(2)

        if mode == 'normal' or mode == 'largebatch':
            loss.mean().backward()
            opt_normal.step()
            opt_normal.zero_grad()
        if mode == 'oracle':
            loss.mean().backward()
            for t, (xt, yt) in enumerate(past_data):
                li = (model(xt).squeeze(1) - yt).pow(2)
                gds = torch.autograd.grad(li.mean(), model.parameters())
                for p, g in zip(model.parameters(), gds):
                    p.grad += beta ** (len(past_data) - t) * g
            past_data = past_data[-horizon+1:] + [(xi, yi)]
            for p in model.parameters():
                p.data -= p.grad * learning_rate
                p.grad.fill_(0)
        elif mode == 'corr':
            df = p2v(torch.autograd.grad(o.mean(), model.parameters(), retain_graph=True))
            dL = p2v(torch.autograd.grad(loss.mean(), model.parameters(), retain_graph=True))
            z = (df)[:, None] * df[None, :]
            u = learning_rate * beta * (Z @ (mu_bar - q))
            q.mul_(beta).add_(u)
            mu_bar.mul_(beta).add_(dL, alpha=dampen)
            Z.mul_(beta).add_(z, alpha=dampen)
            for p, gi in zip(model.parameters(), v2p(mu_bar - q)):
                p.data.add_(gi, alpha=-learning_rate)
        elif mode == 'corr_2':
            df = p2v(torch.autograd.grad(o.mean(), model.parameters(), retain_graph=True))
            dL = p2v(torch.autograd.grad(loss.mean(), model.parameters(), retain_graph=True))
            z = (df)[:, None] * df[None, :]
            u = learning_rate * beta * (Z @ (mu_bar + q))
            q.mul_(beta).add_(u)
            mu_bar.mul_(beta).add_(dL, alpha=dampen)
            Z.mul_(beta).add_(z, alpha=dampen)
            for p, gi in zip(model.parameters(), v2p(mu_bar + q)):
                p.data.add_(gi, alpha=-learning_rate)
        losses.append(loss.mean().item())
    losses = np.float32(losses)

    conn = lmdb.open(path, map_size=int(16 * 2 ** 30)) # 16gb max?

    with conn.begin(write=True) as txn:
        k = str(hps['id']).encode()
        txn.put(k + b'_losses', gzip.compress(pickle.dumps(losses)))
        txn.put(k + b'_hps', gzip.compress(pickle.dumps(hps)))
        txn.put(k + b'_model', gzip.compress(pickle.dumps(model.state_dict())))



if __name__ == '__main__':
    import ray
    ray.init(num_cpus=10)

    if 1:
        path = 'results/aug_17_sin_reg'

        all_hps = [{'nhid': nhid,
                    'beta': beta,
                    'mbsize': mbsize,
                    'mode': mode,
                    'lr': lr,
                    'seed': seed}
                   for nhid in [8,16,32]
                   for beta in [0.9, 0.99]
                   for lr in [5e-3, 1e-2]
                   for mode in ['normal', 'oracle', 'corr', 'corr_2']
                   for mbsize in [4,16,64]
                   for seed in range(10)]

    for i, u in enumerate(all_hps):
        u['id'] = i
    print(len(all_hps), 'experiments')

    conn = lmdb.open(path, map_size=int(16 * 2 ** 30)) # 16gb max?

    rmain = ray.remote(main)
    with conn.begin() as txn:
        cursor = txn.cursor()
        jobs = [rmain.remote(path, i) for i in all_hps
                if not cursor.set_key(str(i['id']).encode() + b'_hps')]

    print(len(jobs), 'jobs')

    with tqdm.tqdm(total=len(jobs), smoothing=0) as t:
        while len(jobs):
            ready, jobs = ray.wait(jobs)
            for i in ready:
                t.update()
