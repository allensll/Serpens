import os
import sys
import argparse
absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(absPath)
import utils
import mpc


def main():
    parser = argparse.ArgumentParser(description='User')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--k', type=int, default=13)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--servers', type=int, default=2)
    parser.add_argument('--alpha', type=int, default=10)
    args = parser.parse_args()

    batch_size = 64
    if args.dataset == 'mnist':
        _, test_loader = utils.load_MNIST(batch_size, test_batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        _, test_loader = utils.load_CIFAR10(batch_size, test_batch_size=args.batch_size)
    elif args.dataset == 'cifar100':
        _, test_loader = utils.load_CIFAR100(batch_size, test_batch_size=args.batch_size)
    elif args.dataset == 'imagenet':
        _, test_loader = utils.load_ImageNet(batch_size, test_batch_size=args.batch_size)
    else:
        print('Not exist dataset:{}'.format(args.dataset))
        return

    user = mpc.User()
    user.start('127.0.0.1', 14714) # 6006

    sum_correct = 0
    for idx, (data, target) in enumerate(test_loader):
        with utils.timer('{}'.format(idx)):
            xi = utils.partition(data, alpha=args.alpha, n=args.servers)
            user.upload(list(zip(xi, [args.k]*args.servers)))

            output = user.get_res()
            pred = output.argmax(dim=1, keepdim=True)

        correct = pred.eq(target.view_as(pred)).sum().item()
        sum_correct += correct
        print('Test {} {} images: corect: {}\n'.format(args.batch_size, args.dataset, correct))
        if idx == args.iters - 1:
            break
    test_acc = 100. * sum_correct / ((idx+1) * args.batch_size)
    print('Test Acc: {:.2f}%'.format(test_acc))

    user.close()


if __name__ == '__main__':
    main()
