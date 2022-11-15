import os
import argparse
import sys

absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(absPath)
# print(sys.path)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
from models import lenet5


def train(model, device, data_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = F.cross_entropy(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))


def test(model, device, data_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # break

    test_loss /= len(data_loader.dataset)

    test_acc = 100. * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), test_acc))
    return test_acc


def main():
    parser = argparse.ArgumentParser(description='Train LeNet5 for MNIST in PyTorch')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(147)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = utils.load_MNIST(args.batch_size, **kwargs)
    model = lenet5()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())

    cudnn.benchmark = True

    best_acc = 0
    for e in range(1, args.epoch + 1):
        train(model, device, train_loader, optimizer, e)
        acc = test(model, device, test_loader, e)

        # remember best acc
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            if args.save_model:
                torch.save({
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                }, '../pretrained/lenet5.pth')

        if args.save_model:
            torch.save({
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, '../pretrained/lenet5.pth')

    print('Best Acc : {}%'.format(best_acc))


def acc():
    parser = argparse.ArgumentParser(description='Train LeNet5 for MNIST in PyTorch')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cuda', default=False)
    args = parser.parse_args()

    test_batch_size = 1000

    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(147)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = utils.load_MNIST(args.batch_size, test_batch_size=test_batch_size, **kwargs)
    model = lenet5(pretrained=True)
    model = model.to(device)
    model.eval()
    cudnn.benchmark = True

    test_acc = test(model, device, test_loader, 0)
    print(test_acc)


if __name__ == '__main__':

    # main()
    acc()
