import time
import torch
import torch.nn as nn
import numpy as np
import lmdb
import tqdm
import gzip
import pickle
import math

import numpy as np
import pickle
import gzip
import torch
import torch.nn as nn
import numpy as np
import scipy.io


svhn = scipy.io.loadmat('svhn/train_32x32.mat')
svhn_test = scipy.io.loadmat('svhn/test_32x32.mat')

device = torch.device('cuda')
xtrain = torch.tensor(svhn['X'].transpose(3,2,0,1) / 255.).float().to(device)
ytrain = torch.tensor(svhn['y']-1).to(device).long()[:, 0]
xvalid = torch.tensor(svhn_test['X'].transpose(3,2,0,1) / 255.).float().to(device)
yvalid = torch.tensor(svhn_test['y']-1).to(device).long()[:, 0]

avg_colors = xtrain.mean((0, 2, 3))
xtrain -= avg_colors[None, :, None, None]
xvalid -= avg_colors[None, :, None, None]


def main(path, hps):

    seed = 142857 + hps['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    nhid = hps['nhid']
    beta = hps['beta']
    learning_rate = hps['lr']
    act = nn.LeakyReLU()

    mbsize = hps['mbsize']
    mode = hps['mode']
    mbrange = torch.arange(mbsize).to(device)
    mbrangev = torch.arange(512).to(device)




    model = nn.Sequential(nn.Conv2d(3, nhid, 3, 2, 1), act,
                          nn.Conv2d(nhid, nhid * 2, 3, 2, 1), act,
                          nn.Conv2d(nhid * 2, nhid * 2, 3, 2, 1), act,
                          nn.Conv2d(nhid * 2, nhid, 3, 1, 1), act,
                          nn.Flatten(),
                          nn.Linear(nhid * 4 * 4, nhid * 4), act,
                          nn.Linear(nhid * 4, nhid * 4), act,
                          nn.Linear(nhid * 4, 10), nn.LogSoftmax(1))

    for i in model.parameters():
        if len(i.shape) == 1:
            i.data.fill_(0)
        elif len(i.shape) == 2:
            k = np.sqrt(6 / (i.shape[0] + i.shape[1]))
            i.data.uniform_(-k, k)
        elif len(i.shape) == 4:
            k = np.sqrt(6 / (i.shape[0] + i.shape[1] * i.shape[2] * i.shape[3]))
            i.data.uniform_(-k, k)

    model.to(device)
    msgd = torch.optim.SGD(model.parameters(), learning_rate, momentum=beta, dampening=beta)


    paramslices = [0] + list(np.cumsum([np.prod(i.shape) for i in model.parameters()]))
    #print(paramslices[-1],'parameters')
    p2v = lambda x: torch.cat([i.reshape(-1) if i is not None else torch.zeros_like(p).reshape(-1)
                               for i, p in zip(x, model.parameters())])
    v2p = lambda x: [x[u:v].reshape(p.shape)
                     for u,v,p in zip(paramslices[:-1], paramslices[1:], model.parameters())]
    losses = []
    accs = []

    test_losses = []
    test_accs = []
    if mode == 'corr':
        mu_bar = torch.zeros(paramslices[-1], device=device)
        Z = torch.zeros((paramslices[-1], paramslices[-1]), device=device)
        q = torch.zeros(paramslices[-1], device=device)

    horizon = int(np.ceil(2 * 1 / (1-beta)))

    past_data = []

    t0 = time.time()

    for i in tqdm.tqdm(range(20000), leave=False):
        idx = np.random.randint(0, xtrain.shape[0], mbsize)
        xi = xtrain[idx]
        yi = ytrain[idx]
        o = model(xi)
        oy = -o[mbrange, yi]
        loss = oy.mean()
        accuracy = (o.argmax(1) == yi).float().mean()
        losses.append(loss.item())
        accs.append(accuracy.item())

        if mode == 'normal':
            loss.backward()
            msgd.step()
            msgd.zero_grad()
        elif mode == 'oracle':
            (loss * (1-beta)).backward()
            if len(past_data):
                xs = torch.cat([i[0] for i in past_data], 0)
                ys = torch.cat([i[1] for i in past_data], 0)
                b = torch.cat([torch.ones(mbsize) * (1-beta) * beta ** (len(past_data) - t)
                               for t in range(len(past_data))], 0).to(device)
                o = model(xs)
                loss = -o[torch.arange(ys.shape[0], device=device), ys].mul(b).sum() / mbsize
                loss.backward()
            past_data = past_data[-horizon+1:] + [(xi, yi)]
            for p in model.parameters():
                p.data -= p.grad * learning_rate
                p.grad.fill_(0)
        elif mode == 'corr':
            df = p2v(torch.autograd.grad(oy.mean(), model.parameters(), retain_graph=True))
            dL = p2v(torch.autograd.grad(loss.mean(), model.parameters(), retain_graph=True))
            if False:
                ddvt = p2v(torch.autograd.grad((delta.detach() * vt).mean(), model.parameters(), create_graph=True))
                dHf = torch.stack([
                    p2v(torch.autograd.grad(ddvt[i], model.parameters(), retain_graph=True, allow_unused=True))
                    for i in range(ddvt.shape[0])
                ])
            else:
                dHf = 0
            z1 = (df)[:, None] * df[None, :]
            z = z1 + dHf
            u = learning_rate * beta * (Z @ (mu_bar - q))
            q.mul_(beta).add_(u)
            mu_bar.mul_(beta).add_(dL, alpha=1-beta)
            Z.mul_(beta).add_(z, alpha=1-beta)
            for p, gi in zip(model.parameters(), v2p(mu_bar - q)):
                p.data.add_(gi, alpha=-learning_rate)

        if not i % 100:
            idx = np.random.randint(0, xvalid.shape[0], 512)
            xi = xvalid[idx]
            yi = yvalid[idx]
            with torch.no_grad():
                o = model(xi)
            loss = -o[mbrangev, yi].mean()
            accuracy = (o.argmax(1) == yi).float().mean()
            test_losses.append(loss.item())
            test_accs.append(accuracy.item())
            #print(i, time.time()-t0, loss.item(), accuracy.item())
            t0 = time.time()


    conn = lmdb.open(path, map_size=int(16 * 2 ** 30)) # 16gb max?
    with conn.begin(write=True) as txn:
        k = str(hps['id']).encode()
        txn.put(k + b'_losses', gzip.compress(pickle.dumps(np.float32(losses))))
        txn.put(k + b'_accs', gzip.compress(pickle.dumps(np.float32(accs))))
        txn.put(k + b'_test_losses', gzip.compress(pickle.dumps(np.float32(test_losses))))
        txn.put(k + b'_test_accs', gzip.compress(pickle.dumps(np.float32(test_accs))))
        txn.put(k + b'_model', gzip.compress(pickle.dumps(model.state_dict())))
        txn.put(k + b'_hps', gzip.compress(pickle.dumps(hps)))


if __name__ == '__main__':

    path = 'results/aug_14_svhn'

    all_hps = [{'nhid': nhid,
                'beta': beta,
                'mbsize': mbsize,
                'mode': mode,
                'lr': lr,
                'seed': seed}
               for seed in range(5)
               for nhid in [8,16]
               for beta in [0.9, 0.99]
               for lr in [5e-3, 1e-2]
               for mode in ['normal', 'oracle', 'corr']
               for mbsize in [4,16,64]]

    for i, u in enumerate(all_hps):
        u['id'] = i
    print(len(all_hps), 'experiments')

    import os
    exp_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    main(f'results/aug_17_svhn_sub/e_{exp_id}/',
         all_hps[exp_id])
