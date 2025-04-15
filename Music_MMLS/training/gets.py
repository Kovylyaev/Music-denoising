from torch import nn, optim

from Music_MMLS.models.unet import UNet

models = {
    '': lambda: None,
    'UNet': UNet,
}

criterion = {
    '': lambda: None,
    'MSE': nn.MSELoss,
}

optimizers = {
    '': lambda: None,
    'Adam': optim.Adam,
}

scheduler = {
    '': lambda: None,
}


def get_model(model_name: str = 'UNet'):
    return models[model_name]

def get_criterion(criterion_name: str = 'MSE'):
    return criterion[criterion_name]

def get_optimizer(optimizer_name: str = 'Adam'):
    return optimizers[optimizer_name]

def get_scheduler(scheduler_name: str = ''):
    return scheduler[scheduler_name]