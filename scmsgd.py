import torch
import numpy as np

_verbose = False

# Staleness-Corrected Momentum-SGD
class SCMSGD:

    def __init__(self, parameters, alpha, momentum=0, dampening=True, weight_decay=0,
                 diagonal=False, beta2=0, epsilon=1e-8):
        """
        parameters: list of parameters
        alpha: learning rate
        momentum: momentum rate, beta (beta_1 in Adam)
        dampening: multiply gradient accumulation by (1-dampening), or if True, dampening=beta
        weight_decay: L2 weight decay factors
        diagonal: if True uses the diagonal approximation correction, requires 3n parameters,
          otherwise uses the full matrix, 2n + n**2 parameters.
        beta2: Adam beta_2
        epsilon: Adam stability parameter
        """

        self.parameters = list(parameters)
        self.device = self.parameters[0].device
        self.alpha = alpha
        self.beta = momentum
        self.dampening = 0 if dampening is False else (momentum if dampening is True else dampening)
        self.weight_decay = weight_decay
        self.diagonal = diagonal
        self.beta2 = beta2
        self.epsilon = epsilon

        # Slice indices within a concatenated vector ofthe parameters
        self.slices = [0] + list(np.cumsum([np.prod(i.shape) for i in self.parameters]))
        # Parameter list to vector
        self.p2v = lambda x: torch.cat([
            i.reshape(-1) if i is not None else torch.zeros_like(p).reshape(-1)
            for i, p in zip(x, self.parameters)])

        # Vector to parameter list
        self.v2p = lambda x: [
            x[u:v].reshape(p.shape)
            for u,v,p in zip(self.slices[:-1], self.slices[1:], self.parameters)]

        # Tracking values
        self.mu = torch.zeros(self.slices[-1], device=self.device)
        self.zeta = (torch.zeros((self.slices[-1], self.slices[-1]), device=self.device)
                     if not diagonal else
                     torch.zeros((self.slices[-1],), device=self.device))
        self.eta = torch.zeros(self.slices[-1], device=self.device)
        self.tracking_parameters = [self.mu, self.zeta, self.eta]
        if beta2 > 0:
            self.v = torch.zeros(self.slices[-1], device=self.device)
            self.tracking_parameters += [self.v]

        self._step = 0

        if _verbose:
            print(f"Model has {self.slices[-1]:_} parameters")
            print("Currently tracking",
                  f"{sum(np.prod(i.shape) for i in self.tracking_parameters):_}",
                  "extra parameters")
            print("Block diagonal would require",
                  f"sum({[np.prod(i.shape)**2 for i in self.parameters]}) =",
                  f"{sum([np.prod(i.shape)**2 for i in self.parameters]):_}",
                  'extra parameters')

    def set_predictions(self, v, gvp=None):
        """Set predictions to be used for correction

        e.g. for TD(0), set:
        v = V(s), gvp = gamma * V(s')

        It is up to the caller to ensure that gvp is multiplied by 0
        if s is terminal. For a minibatch, simply pass the mean (or
        sum if doing loss.sum(mbdim)).

        If gvp is None, then this is just supervised learning, zeta is
        computed as the outer product of gV with itself.

        """
        self._v = v
        self._gvp = gvp

    def step(self, return_grads=False):
        dvt = self.p2v(torch.autograd.grad(self._v, self.parameters, retain_graph=True))
        dvtp = (self.p2v(torch.autograd.grad(self._gvp, self.parameters, retain_graph=True))
                if self._gvp is not None else 0)
        dL = self.p2v([i.grad for i in self.parameters])
        if self.weight_decay:
            # We ignore weight decay when computing the correction
            # This should not be problematic but further testing required
            dL.add_(self.p2v(self.parameters).detach(), alpha=self.weight_decay)

        # The update
        # Recompute \mu_{t-1}
        mu_tm1 = self.mu - self.eta

        # Update eta
        if self.diagonal:
            z = (dvt - dvtp) * dvt # diagonal of outer product
            self.eta.mul_(self.beta).add_(self.alpha * self.beta * (self.zeta * mu_tm1)) # eta
        else:
            z = (dvt - dvtp)[:, None] * dvt[None, :] # outer product
            self.eta.mul_(self.beta).add_(self.alpha * self.beta * (self.zeta.T @ mu_tm1)) # eta
        # Update zeta
        self.zeta.mul_(self.beta).add_(z, alpha=1-self.dampening)
        # Update mu
        self.mu.mul_(self.beta).add_(dL, alpha=1-self.dampening)

        # Compute corrected momentum \mu_t
        mu_t = self.mu - self.eta
        if not self._step % 100 and False:
            print(self._step, abs(self.eta).mean().item())
            print(dvt)
            print(dvtp)

        # second order moment
        self._step += 1
        if self.beta2 > 0:
            self.v.mul_(self.beta2).addcmul_(dL, dL, value=1-self.beta2)
            vc = self.v.sqrt().div_(np.sqrt(1 - self.beta2**self._step)).add_(self.epsilon)
            # Update parameters
            for p, g, d in zip(self.parameters, self.v2p(mu_t), self.v2p(vc)):
                p.data.addcdiv_(g, d, value=-(self.alpha / (1-self.beta**self._step)))
        # no second order moment
        else:
            # Update parameters
            for p, g in zip(self.parameters, self.v2p(mu_t)):
                p.data.add_(g, alpha=-self.alpha)
        if return_grads:
            return self.v2p(mu_t), z, dL

    def zero_grad(self):
        for i in self.parameters:
            if i.grad is not None:
                i.grad.fill_(0)



class OptChain:

    def __init__(self, opts):
        self.opts = opts

    def set_predictions(self, v, gvp):
        for o in self.opts:
            if hasattr(o, 'set_predictions'):
                o.set_predictions(v, gvp)

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()




import torch
from torch.optim.optimizer import Optimizer, required


class FixedSGD(Optimizer):
    """Fix SGD so that its momentum is a fair comparison to ours (see code)"""

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FixedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FixedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        buf.mul_(1 - dampening) # This is the fix
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
