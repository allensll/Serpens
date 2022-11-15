import os
import sys
import argparse
absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(absPath)
import utils

import mpc


def main():
    parser = argparse.ArgumentParser(description='Server')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--ipu', type=str, default='127.0.0.1')
    parser.add_argument('--portu', type=int, default=14714)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        model = 'lenet5'
    elif args.dataset == 'cifar10':
        model = 'minionn'
    elif args.dataset == 'cifar100':
        model = 'resnet32'
    elif args.dataset == 'imagenet':
        model = 'resnet50'
    else:
        print('Not exist dataset:{}'.format(args.dataset))
        return

    srv = mpc.Server(args.dataset, model, m=1)

    srv.connect(args.ipu, args.portu)
    for i in range(args.iters):
        with utils.timer(i):
            srv.inference()
    srv.close()


if __name__ == '__main__':
    main()
