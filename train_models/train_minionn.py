import os
import argparse
import sys

absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(absPath)
# print(sys.path)

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
from models import minionn


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
    parser = argparse.ArgumentParser(description='Train MiniONN for CIFAR10 in PyTorch')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()

    lr = 0.01
    m = 0.9
    wd = 0.0001

    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(147)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = utils.load_CIFAR10(args.batch_size, **kwargs)
    model = minionn()
    model = model.to(device)
    optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': lr}], lr, momentum=m, weight_decay=wd)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    cudnn.benchmark = True

    best_acc = 0
    for e in range(1, args.epoch + 1):
        train(model, device, train_loader, optimizer, e)
        acc = test(model, device, test_loader, e)
        lr_scheduler.step()

        # remember best acc
        is_best = acc > best_acc
        if is_best:
            best_acc = acc
            if args.save_model:
                torch.save({
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                }, '../pretrained/minionn/checkpoint_best.pth')

        if args.save_model:
            torch.save({
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, '../pretrained/minionn/model.pth')

    print('Best Acc : {}%'.format(best_acc))


def acc():
    parser = argparse.ArgumentParser(description='Train MiniONN for CIFAR10 in PyTorch')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()

    test_batch_size = 1000

    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(147)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = utils.load_CIFAR10(args.batch_size, test_batch_size=test_batch_size, **kwargs)
    model = minionn(pretrained=True)
    model = model.to(device)
    model.eval()
    cudnn.benchmark = True

    acc = test(model, device, test_loader, 0)
    print(acc)


if __name__ == '__main__':

    # main()
    acc()
