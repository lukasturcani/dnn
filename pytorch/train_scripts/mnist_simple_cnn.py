import argparse
import logging
import torch.nn.functional as F
import torch
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from dnn_models.pytorch.models.simple_cnn import SimpleCNN


logger = logging.getLogger(__name__)


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_id % args.log_interval == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
            msg = msg.format(
                epoch,
                batch_id * len(data),
                len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                loss.item())
            logger.info(msg)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            test_loss += F.nll_loss(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    msg = ('\nTest set: Average loss: {:.4f}, '
           'Accuracy: {}/{} ({:.0f}%)\n')
    msg = msg.format(test_loss,
                     correct,
                     len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset))
    logger.info(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--conv_in_channels',
                        default=[1, 20],
                        nargs='+',
                        type=int)
    parser.add_argument('--conv_out_channels',
                        default=[20, 50],
                        nargs='+',
                        type=int)
    parser.add_argument('--conv_kernel_sizes',
                        default=[5, 5],
                        nargs='+',
                        type=int)
    parser.add_argument('--conv_strides',
                        default=[1, 1],
                        nargs='+',
                        type=int)
    parser.add_argument('--conv_paddings',
                        default=[0, 0],
                        nargs='+',
                        type=int)
    parser.add_argument('--conv_dilations',
                        default=[1, 1],
                        nargs='+',
                        type=int)
    parser.add_argument('--pool_kernel_sizes',
                        default=[2, 2],
                        nargs='+',
                        type=int)
    parser.add_argument('--pool_strides',
                        default=[2, 2],
                        nargs='+',
                        type=int)
    parser.add_argument('--pool_paddings',
                        default=[0, 0],
                        nargs='+',
                        type=int)
    parser.add_argument('--pool_dilations',
                        default=[1, 1],
                        nargs='+',
                        type=int)
    parser.add_argument('--fc_input_size', default=4*4*50, type=int)
    parser.add_argument('--fcs',
                        default=[500, 10],
                        nargs='+',
                        type=int)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--logging_level',
                        default=logging.DEBUG,
                        type=int)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--database_root',
                        default='/home/lukas/databases')

    args = parser.parse_args()

    logging_fmt = ('%(asctime)s - %(levelname)s - '
                   '%(module)s - %(msg)s')
    date_fmt = '%d-%m-%Y %H:%M:%S'
    logging.basicConfig(level=args.logging_level,
                        format=logging_fmt,
                        datefmt=date_fmt)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_mnist = datasets.MNIST(root=args.database_root,
                                 train=True,
                                 download=True,
                                 transform=transform)

    train_loader = DataLoader(
                        dataset=train_mnist,
                        batch_size=args.train_batch_size,
                        shuffle=True,
                        num_workers=1,
                        pin_memory=True)

    test_mnist = datasets.MNIST(root=args.database_root,
                                train=False,
                                transform=transform)
    test_loader = DataLoader(
                      dataset=test_mnist,
                      batch_size=args.test_batch_size,
                      shuffle=True,
                      num_workers=1,
                      pin_memory=True)

    model = SimpleCNN(conv_in_channels=args.conv_in_channels,
                      conv_out_channels=args.conv_out_channels,
                      conv_kernel_sizes=args.conv_kernel_sizes,
                      conv_strides=args.conv_strides,
                      conv_paddings=args.conv_paddings,
                      conv_dilations=args.conv_dilations,
                      pool_kernel_sizes=args.pool_kernel_sizes,
                      pool_strides=args.pool_strides,
                      pool_paddings=args.pool_paddings,
                      pool_dilations=args.pool_dilations,
                      fc_input_size=args.fc_input_size,
                      fcs=args.fcs)

    model.to('cuda')
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, test_loader)


if __name__ == '__main__':
    main()
