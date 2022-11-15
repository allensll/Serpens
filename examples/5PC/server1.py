import os
import sys
import argparse
absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(absPath)
import utils
import mpc


def main():
    parser = argparse.ArgumentParser(description='Server 1')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--ip0', type=str, default='127.0.0.1')
    parser.add_argument('--port0', type=int, default=20202)
    parser.add_argument('--ipu', type=str, default='127.0.0.1')
    parser.add_argument('--portu', type=int, default=14714)
    parser.add_argument('--iters', type=int, default=7)
    parser.add_argument('--servers', type=int, default=4)
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

    srv = mpc.Server(args.dataset, model, relu='relumpc', pool='maxpoolmpc', t='host',
                     agtaddr=args.ip0, agtport=args.port0, m=args.servers)
    # connect to user
    srv.connect(args.ipu, args.portu) # 58.213.25.18 XXXXX
    for i in range(args.iters):
        with utils.timer('total'):
            srv.inference()
    srv.close()


if __name__ == '__main__':
    main()
