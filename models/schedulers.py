from torch.optim.lr_scheduler import LambdaLR
import math
from functools import partial


def _cosine_scheduler(current_step, total_steps, num_warmup_steps, base_lr, min_lr=0):
    if current_step < num_warmup_steps:
        return current_step/num_warmup_steps
    
    progress = min(1, (current_step - num_warmup_steps) / (total_steps - num_warmup_steps))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return (min_lr / base_lr) + (base_lr - min_lr) * cosine_decay

def _trap_scheduler(current_step, total_steps, num_warmup_steps, base_lr, min_lr=0):
    if current_step < num_warmup_steps:
        return current_step/num_warmup_steps

    if current_step < int(0.85 * total_steps):
        return 1
    
    if current_step >= total_steps:
        return min_lr/base_lr
    
    decay_steps = total_steps - int(0.85 * total_steps)
    progress = (current_step - int(0.85 * total_steps))/decay_steps
    trap_decay = 1.0 - progress
    return (min_lr/base_lr) + (1 - (min_lr/base_lr)) * trap_decay


def get_cosine_scheduler(optimizer, total_steps, num_warmup_steps, base_lr, min_lr=0, last_epoch=-1):
    lr_lambda = partial(
        _cosine_scheduler,
        total_steps=total_steps,
        num_warmup_steps=num_warmup_steps,
        base_lr=base_lr,
        min_lr=min_lr
    )
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

def get_trap_scheduler(optimizer, total_steps, num_warmup_steps, base_lr, min_lr=0, last_epoch=-1):
    lr_lambda = partial(
        _trap_scheduler,
        total_steps=total_steps,
        num_warmup_steps=num_warmup_steps,
        base_lr=base_lr,
        min_lr=min_lr
    )
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
