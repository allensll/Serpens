import pickle
import socket
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F

import torch
import torch.nn as nn

import nl2pc

from .utils import timer, recv_object, send_object
from .layers import ReLU2PC, ReLUMPC, MaxPool2d2PC, MaxPool2dMPC
from models import *
# from models import lenet5


class Agent:
    def __init__(self):
        self.clients = list()
        self.rand_seed = None

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, ip, port):
        self.socket.connect((ip, port))
        print('---Agent: connected')

    def close(self):
        for clt in self.clients:
            clt.close()
        self.socket.close()
        print("---Agent: close connect")

    def recv_input(self):
        return recv_object(self.socket)

    def send_output(self, x):
        sl = send_object(self.socket, x)
        # print("---Agent: send output")
        return sl


class Server:
    def __init__(self, dataset, model, relu='relu', pool='maxpool', t=None, agtaddr='127.0.0.1', agtport='20202', m=2, nthreads=1):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dataset = dataset
        self.model_name = model
        self.relu = relu
        self.t = t
        self.m = m
        if dataset == 'mnist' or dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'imagenet':
            num_classes = 1000
        else:
            raise ValueError('No dataset: {}'.format(dataset))

        if t:
            addr = agtaddr
            port = agtport
            if t == 'host':
                self.agt = Agent()
                self.agt.connect(addr, port)
            elif t == '2pc_s':
                self.agt = nl2pc.Create(nl2pc.ckks_role.SERVER, address=addr, port=port, nthreads=nthreads)
            elif t == '2pc_c':
                self.agt = nl2pc.Create(nl2pc.ckks_role.CLIENT, address=addr, port=port, nthreads=nthreads)
            else:
                raise ValueError('(host, 2pc_s, 2pc_c), but get {}'.format(t))

            if relu == 'relu':
                relu = nn.ReLU()
            elif relu == 'relu2pc':
                relu = ReLU2PC(self.agt)
            elif relu == 'relumpc':
                relu = ReLUMPC(self.agt)
            else:
                raise ValueError('relu or relumpc, but get {}'.format(relu))

            if model in ['lenet5']:
                if pool == 'avgpool':
                    pool = nn.AvgPool2d(2)
                elif pool == 'maxpool':
                    pool = nn.MaxPool2d(2)
                elif pool == 'maxpool2pc':
                    pool = MaxPool2d2PC(2, agt=self.agt)
                elif pool == 'maxpoolmpc':
                    pool = MaxPool2dMPC(2, agt=self.agt)
                else:
                    raise ValueError('avgpool or maxpool or maxpoolmpc, but get {}'.format(pool))
                self.model = eval(model)(relu, pool, pretrained=True, n=m)
            elif model in ['minionn']:
                self.model = eval(model)(relu, pretrained=True, n=m)
            elif model in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']:
                self.model = eval(model)(relu, num_classes=num_classes, pretrained=True, n=m)
            elif model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                if pool == 'avgpool':
                    pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                elif pool == 'maxpool':
                    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif pool == 'maxpool2pc':
                    pool = MaxPool2d2PC(kernel_size=3, stride=2, padding=1, agt=self.agt)
                elif pool == 'maxpoolmpc':
                    pool = MaxPool2dMPC(kernel_size=3, stride=2, padding=1, agt=self.agt)
                else:
                    raise ValueError('avgpool or maxpool or maxpoolmpc, but get {}'.format(pool))
                self.model = eval(model)(relu, pool, pretrained=True, n=m)
            elif model in ['densenet21', 'densenet41', 'densenet121', 'densenet201']:
                if pool == 'avgpool':
                    pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                elif pool == 'maxpool':
                    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif pool == 'maxpool2pc':
                    pool = MaxPool2d2PC(kernel_size=3, stride=2, padding=1, agt=self.agt)
                elif pool == 'maxpoolmpc':
                    pool = MaxPool2dMPC(kernel_size=3, stride=2, padding=1, agt=self.agt)
                else:
                    raise ValueError('avgpool, maxpool, maxpool2d2pc or maxpoolmpc, but get {}'.format(pool))
                self.model = eval(model)(relu, pool, num_classes=num_classes, pretrained=True, n=m)
                # self.model = eval(model)(relu, pool, pretrained=True, n=m)
            else:
                raise ValueError('Not exist model:{}'.format(model))
        else:
            self.model = eval(model)(num_classes=num_classes, pretrained=True)
        # self.model = model.to(torch.device('cpu'))
        self.model.eval()

    def connect(self, ip, port):
        self.socket.connect((ip, port))
        print('---Server: connected')

    def close(self):
        if self.t == 'host':
            self.agt.close()
        self.socket.close()
        print("---Server: close connect")

    def inference(self):
        x = recv_object(self.socket)
        if self.t == 'host':
            if isinstance(x, tuple):
                x, self.agt.rand_seed = x
                torch.manual_seed(self.agt.rand_seed)
            else:
                raise ValueError('Input should be a tuple (x, seed)')

        print('---Server: start inference')
        y = self.model(x)
        print('---Server: finish inference')
        send_object(self.socket, y)


class ServerCentral:
    """Used in MPC.

    """
    def __init__(self, dataset, model, relu='relu', pool='maxpool', m=2):
        self.model_name = model
        self.m = m
        self.clients = list()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # the number of relu + maxpool layers
        relu_dict = {
            'lenet5': 3,
            'minionn': 7,
            'resnet20': 19, 'resnet32': 31, 'resnet44': 43, 'resnet56': 55, 'resnet110': 109, 'resnet1202': 1201,
            'resnet18': 17, 'resnet34': 33, 'resnet50': 49, 'resnet101': 100, 'resnet152': 151,
            # small
            'densenet21': 19, 'densenet41': 39, 'densenet121': 119, 'densenet201': 199,
            # large = small + 1 + 1
        }
        maxpool_dict = {
            'lenet5': 2,
            'minionn': 0,
            'resnet20': 0, 'resnet32': 0, 'resnet44': 0, 'resnet56': 0, 'resnet110': 0, 'resnet1202': 0,
            'resnet18': 1, 'resnet34': 1, 'resnet50': 1, 'resnet101': 1, 'resnet152': 1,
            'densenet21': 1, 'densenet41': 1, 'densenet121': 1, 'densenet201': 1,
        }
        self.l = 0
        if relu == 'relumpc':
            self.l += relu_dict[model]
        if pool == 'maxpoolmpc':
            self.l += maxpool_dict[model]
        print('--model layer {}'.format(self.l))

    def start(self, ip, port):
        self.socket.bind((ip, port))
        print('---ServerCentral: started {}: {}'.format(ip, port))
        self.socket.listen(1)
        print('---ServerCentral: wait...')
        for i in range(self.m):
            client, address = self.socket.accept()
            self.clients.append(client)
            print('---ServerCentral: add client {} : {}'.format(i, address))

    def close(self):
        for clt in self.clients:
            clt.close()
        self.socket.close()
        print("---ServerCentral: close connect")

    def recv_input(self):
        res = list()
        tag = list()
        for i in range(self.m):
            t, r = recv_object(self.clients[i])
            tag.append(t)
            res.append(r.float())
        assert len(set(tag)) == 1
        return tag[0], res

    def send_output(self, xs):
        sl = 0
        for i in range(self.m):
            sl += send_object(self.clients[i], xs[i])
        return sl

    def inference(self):
        for i in range(self.l):
            print(i)
            # 2 compute
            tag, xs = self.recv_input()
            with timer('central'):
                if tag == 'relu':
                    a = sum(xs)
                    F.relu(a, inplace=True)
                    xs[self.m-1] = a
                    for i in range(self.m-1):
                        xs[i] = torch.rand_like(xs[i])-0.5
                        xs[self.m-1] -= xs[i]
                elif tag == 'maxpool':
                    a = sum(xs)
                    xs[self.m-1], idx = torch.max(a, dim=1)
                    for i in range(self.m-1):
                        xs[i] = torch.rand_like(xs[self.m-1])-0.5
                        xs[self.m-1] -= xs[i]
                else:
                    raise ValueError('relu or maxpool, but get {}'.format(tag))
                for i in range(self.m):
                    xs[i] = xs[i].half().detach()
            sl = self.send_output(xs)
            # print('send {:.2f}MB'.format(sl/1014.0/1024.0))

        print("---ServerCentral: finish inference")


class User:
    def __init__(self, n_srv=2):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.n_srv = n_srv
        self.servers = list()

    def start(self, ip, port):
        self.socket.bind((ip, port))
        print('---User: started {}: {}'.format(ip, port))
        self.socket.listen(1)
        print('---User: waiting...')
        for i in range(self.n_srv):
            server, address = self.socket.accept()
            self.servers.append(server)
            print('---User: add server {} : {}'.format(i, address))

    def upload(self, data):
        sl = 0
        for i in range(self.n_srv):
            sl += send_object(self.servers[i], data[i])
        print("---User: finish upload send {:.2f}MB".format(sl/1014.0/1024.0))

    def get_res(self):
        res = list()
        for i in range(self.n_srv):
            res.append(recv_object(self.servers[i]))
        print("---User: get result")
        # res = res[0]
        res = sum(res)
        return res

    def close(self):
        for svr in self.servers:
            svr.close()
        self.socket.close()
        print("---User: close connect")

