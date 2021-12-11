
# Adam Optimizer that will work on Shared States

import math
import torch
import torch.optim as optim


class SharedAdam(optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        # param_groups contain all the attributes of the optimizer, 
        # among these attributes are the parameters that we have to optimize
        # these parameters represent the weights of the network contained in self.param_groups['params']
        for group in self.param_groups:
            # for each tensor of weights that we want to optiize
            for p in group['params']:
                # the update made by the adam optimizer is based on: 
                state = self.state[p]
                state['step'] = torch.zeros(1)
                # exponential moving average of the gradient, of moment 1, or order 1
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                # exponential moving average of the square of the gradient, of moment 2, or order 2
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
    
    # accelerator for tensor like CUDA
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                # send computations to a part of GPU or CPU that is accessible to all paralellized thread
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
    
    # step method of Adam optimizer
    # super(SharedAdam, self).step()
    def step(self): 
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss

