import sys
import torch
import click
import json
import datetime
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
import torchvision
import types

from tensorboardX import SummaryWriter

import losses
import datasets
import metrics
import model_selector
from min_norm_solvers import MinNormSolver
from min_norm_solvers import gradient_normalizers

NUM_EPOCHS = 100

@click.command()
@click.option('--param_file', default='params.json', help='JSON parameters file')
def load_trained_model(param_file):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)

    exp_identifier = []
    for (key, val) in params.items():
        if 'tasks' in key:
            continue
        exp_identifier+= ['{}={}'.format(key,val)]

    exp_identifier = '|'.join(exp_identifier)
    params['exp_id'] = exp_identifier

    train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    for m in model:
        model_params += model[m].parameters()

    if 'RMSprop' in params['optimizer']:
        optimizer = torch.optim.RMSprop(model_params, lr=params['lr'])
    elif 'Adam' in params['optimizer']:
        optimizer = torch.optim.Adam(model_params, lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model_params, lr=params['lr'], momentum=0.9)

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']

    save_model_path = r"/mnt/raid/data/chebykin/saved_models/first_model_epoch_100.pkl"
    # "optimizer=Adam|batch_size=170|lr=0.0005|dataset=celeba|normalization_type=loss+|algorithm=mgda|use_approximation=True|scales={'\''0'\'': 0.025, '\''1'\'': 0.025, '\''2'\'': 0.025, '\''3'\'': 0.025, '\''4'\'': 0.025, '\''5'\'': 0.025, '\''6_100_model.pkl"
    state = torch.load(save_model_path)
    # print(state['model_0'])
    # state = {'epoch': epoch + 1,
    #          'model_rep': model['rep'].state_dict(),
    #          'optimizer_state': optimizer.state_dict()}
    #
    model['rep'].load_state_dict(state['model_rep'])
    for t in tasks:
        key_name = 'model_{}'.format(t)
        model[t].load_state_dict(state[key_name])

    for m in model:
        model[m].eval()

    with torch.no_grad():
        for batch_val in val_loader:
            val_images = Variable(batch_val[0].cuda())
            labels_val = {}

            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels_val[t] = batch_val[i + 1]
                labels_val[t] = Variable(labels_val[t].cuda())

            val_rep, _ = model['rep'](val_images, None)
            for t in tasks:
                out_t_val, _ = model[t](val_rep, None)
                print(out_t_val.shape)

            break
if __name__ == '__main__':
    load_trained_model()
