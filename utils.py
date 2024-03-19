import torch
import numpy as np
from scipy import stats
import os
import random
import logging


def make_log_ddp():
    if not os.path.exists('log'):
        os.mkdir('log')
    return len(os.listdir('log'))


def pcc(pred, label):
    """pearson's correlations coefficient"""
    y, x = label, pred
    y_mean = y.mean(dim=0)
    x_mean = x.mean(dim=0)
    rho = ((x*y).mean(dim=0) - x_mean*y_mean) / (((x**2).mean(dim=0) -
                                                  x_mean**2)*((y**2).mean(dim=0) - y_mean**2)).sqrt()
    return rho.mean()


def calc_pearsons(preds, labels):
    r = stats.pearsonr(preds, labels)
    return r[0]


def mean_pearsons(preds, labels):
    # preds = np.row_stack([np.array(p) for p in preds])
    # labels = np.row_stack([np.array(l) for l in labels])
    num_classes = preds.shape[1]

    # ns = {"preds": preds, "labels": labels}
    # np.save('ns.npy', ns)

    class_wise_r = np.array(
        [calc_pearsons(preds[:, i], labels[:, i]) for i in range(num_classes)])
    mean_r = np.mean(class_wise_r)
    return mean_r


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
