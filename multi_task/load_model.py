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
# from min_norm_solvers import MinNormSolver
# from min_norm_solvers import gradient_normalizers

# import model_selector

@click.command()
@click.option('--param_file', default='sample.json', help='JSON parameters file')
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

    # train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
    # loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    tst_loader = datasets.get_test_dataset(params, configs)

    model = model_selector.get_model(params)

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']

    # save_model_path = r"/mnt/raid/data/chebykin/saved_models/first_model_epoch_100.pkl"
    # "optimizer=Adam|batch_size=170|lr=0.0005|dataset=celeba|normalization_type=loss+|algorithm=mgda|use_approximation=True|scales={'\''0'\'': 0.025, '\''1'\'': 0.025, '\''2'\'': 0.025, '\''3'\'': 0.025, '\''4'\'': 0.025, '\''5'\'': 0.025, '\''6_100_model.pkl"'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/model_25nov_epoch31.pkl'
    # save_model_path = r'/mnt/raid/data/chebykin/saved_models/11_50_on_November_27/ep1.pkl'
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/12_21_on_November_27/optimizer=Adam|batch_size=256|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=no_smart_gradient_stuff|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0._1_model.pkl'
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/20_15_on_December_04/optimizer=Adam|batch_size=170|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=none|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0.025|__5___0.025|__6_7_model.pkl'
    # val : test
    # epoch 4, 8.84 : 9.28
    # epoch 7, 8.79 : 9.17
    save_model_path = r'/mnt/raid/data/chebykin/saved_models/05_21_on_December_05/' \
                      r'optimizer=Adam|batch_size=170|lr=0.0005|dataset=celeba|normalization_type=none|algorithm=none|use_approximation=True|scales={_0___0.025|__1___0.025|__2___0.025|__3___0.025|__4___0.025|__5___0.025|__6_' \
                      r'100_model.pkl'
    # val : test
    # epoch 4, 9.02:9.46
    # epoch 7, 8.85:9.41
    # epoch 100, 9.66:9.98
    state = torch.load(save_model_path)

    # print(state['model_0'])
    # state = {'epoch': epoch + 1,
    #          'model_rep': model['rep'].state_dict(),
    #          'optimizer_state': optimizer.state_dict()}
    #

    #!!!!!!
    model['rep'].load_state_dict(state['model_rep'])

    for t in tasks:
        key_name = 'model_{}'.format(t)
        model[t].load_state_dict(state[key_name])

    for m in model:
        model[m].eval()
    j = 0
    with torch.no_grad():
        for batch_val in tst_loader:
            print(j)
            j+=1
            val_images = Variable(batch_val[0].cuda())
            labels_val = {}

            for i, t in enumerate(all_tasks):
                if t not in tasks:
                    continue
                labels_val[t] = batch_val[i + 1]
                labels_val[t] = Variable(labels_val[t].cuda())

            val_rep, _ = model['rep'](val_images, None)
            for i, t in enumerate(tasks):
                out_t_val, _ = model[t](val_rep, None)
                # loss_t = loss_fn[t](out_t_val, labels_val[t])
                # tot_loss['all'] += loss_t.item()
                # tot_loss[t] += loss_t.item()
                metric[t].update(out_t_val, labels_val[t])
                # print(out_t_val)
                # print(labels_val[t])
                # print(metric[t].get_result())
    error_sum = 0
    for t in tasks:
        metric_results = metric[t].get_result()

        for metric_key in metric_results:
            print(f'Task = {t}, acc = {metric_results[metric_key]}')
            error_sum += 1 - metric_results[metric_key]

        metric[t].reset()

    error_sum /= float(len(tasks))
    print( error_sum * 100)

if __name__ == '__main__':
    load_trained_model()
