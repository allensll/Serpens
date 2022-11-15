import os
import time
import shutil
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from scipy import io


data_path = os.path.join(os.path.dirname(__file__), 'data')
torch.random.manual_seed(147)

# tmp_x = None


def partition(data, alpha=10, n=2):
    # x1 = torch.rand_like(data) * alpha
    # global tmp_x
    xi = list()
    for i in range(n-1):
        # if tmp_x is None:
        #     tmp_x = torch.rand(data.shape)
        # x_r = tmp_x * alpha / (n - 1)
        x_r = torch.rand(data.shape) * alpha / (n-1)
        xi.append(x_r)

    # x1 = torch.rand(data.shape) * alpha
    x_r = data.clone().detach()
    for i in xi:
        x_r -= i
    xi.append(x_r)
    return xi


def recovery(data, alpha, n):
    xn = data
    for i in range(n-1):
        xi = torch.rand(data.shape) * alpha
        xn -= xi
    x = xn + 0.5 * alpha * (n-1)
    return x


def norm(x, t='zscore'):

    if t == 'minmax':
        d = torch.max(x, dim=1)[0] - torch.min(x, dim=1)[0]

        for i in range(x.shape[0]):
            x[i] -= torch.min(x, dim=1)[0][i]
            x[i] /= d[i]
    elif t == 'zscore':
        for i in range(x.shape[0]):
            std, mean = torch.std_mean(x[i])
            x[i] -= mean
            if float(std) == 0:
                x[i] = 0
            else:
                x[i] /= std
    return x


def dist(x1, x2, t='pair'):
    x1 = x1.view(x1.shape[0], -1)
    x2 = x2.view(x1.shape[0], -1)
    # x1 = x1[:, :10]
    # x2 = x2[:, -10:]

    x1 = norm(x1)
    x2 = norm(x2)

    if t == 'cos':
        dist = F.cosine_similarity(x1, x2)
    elif t == 'pair':
        # a = x1[:, 13:14]
        # b = x2[:, 13:14]
        dist = F.pairwise_distance(x1, x2)

    return dist


@contextmanager
def timer(text=''):
    """Helper for measuring runtime"""

    time_start = time.perf_counter()
    yield
    print('---{} time: {:.5f} s'.format(text, time.perf_counter()-time_start))


def load_MNIST(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_SVHN(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(data_path, split='train', download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(data_path, split='test', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_CIFAR10(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_CIFAR100(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5089, 0.4874, 0.4419), (0.2683, 0.2574, 0.2771))
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_ImageNet(batch_size, test_batch_size=100, **kwargs):
    train_loader = None
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=datasets.ImageFolder(
    #         os.path.join(data_path, 'ImageNet', 'train'),
    #         transforms.Compose([
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #         ])),
    #     batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'ImageNet', 'val'),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def id_to_synset(id_):
    # return str(synsets[corr_inv[id_]-1][1][0])
    meta_clsloc_file = "data/ImageNet/devkit/data/meta_clsloc.mat"
    # synsets = io.loadmat(meta_clsloc_file)
    synsets = io.loadmat(meta_clsloc_file)["synsets"][0]
    return str(synsets[id_-1][1][0])


def move_valimg():
    val_gt = 'data/ImageNet/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt'
    val_file = 'data/ImageNet/ImageSets/CLS-LOC/val.txt'
    val_path = 'data/ImageNet/val_raw/'
    dst_val_path = 'data/ImageNet/val'
    blacklist = 'data/ImageNet/devkit/data/ILSVRC2015_clsloc_validation_blacklist.txt'

    f_file = open(val_file, mode='r')
    f_gt = open(val_gt, mode='r')
    f_bl = open(blacklist, mode='r')
    lines_file = f_file.readlines()
    lines_gt = f_gt.readlines()
    lines_bl = f_bl.readlines()
    f_file.close()
    f_gt.close()
    f_bl.close()

    fname = [line.strip().split(' ')[0] for line in lines_file]
    gt = [int(line.strip())for line in lines_gt]
    bl = [int(line.strip()) for line in lines_bl]

    for i, name in enumerate(fname):
        if i in bl:
            continue
        synset = id_to_synset(gt[i])
        dst_path = os.path.join(dst_val_path, synset)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        shutil.copy(os.path.join(val_path, '{}.jpeg'.format(name)), os.path.join(dst_path, '{}.jpeg'.format(name)))
        print(name, gt[i], synset)


if __name__ == '__main__':

    move_valimg()