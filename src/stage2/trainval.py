import os
import time
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append(r'/home/featurize/KPDA/')

from src import pytorch_utils
from src.kpda_parser import KPDA
from src.config import Config
from src.stage2.data_generator import DataGenerator
from src.stage2.cascade_pyramid_network import CascadePyramidNet
from src.stage2.viserrloss import VisErrorLoss
from src.lr_scheduler import LRScheduler


def print_log(epoch, lr, train_metrics, train_time, val_metrics=None, val_time=None, save_dir=None, log_mode=None):
    """输出每个epoch的训练和验证结果"""
    if epoch > 1:
        log_mode = 'a'
    train_metrics = np.mean(train_metrics, axis=0)
    str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
    str1 = 'Train:      time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f' \
           % (train_time, train_metrics[0], train_metrics[1], train_metrics[2])
    print(str0)
    print(str1)
    f = open(save_dir + 'kpt_' + config.clothes + '_train_log.txt', log_mode)
    f.write(str0 + '\n')
    f.write(str1 + '\n')
    if val_time is not None:
        val_metrics = np.mean(val_metrics, axis=0)
        str2 = 'Validation: time %3.2f loss: %2.4f loss1: %2.4f loss2: %2.4f' \
               % (val_time, val_metrics[0], val_metrics[1], val_metrics[2])
        print(str2 + '\n')
        f.write(str2 + '\n\n')
    f.close()


def train(data_loader, net, loss, optimizer, lr):
    """训练模型并返回训练结果和训练时间

    :param data_loader: 数据加载器
    :param net: 父类nn.Module 网络模型
    :param loss: 父类nn.Module 损失函数
    :param optimizer: 父类torch.optim.SGD 优化器
    :param lr: 学习率"""
    start_time = time.time()

    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (data, heatmaps, vismaps) in enumerate(data_loader):
        print("i:", i)
        s = time.time()
        # data = data.to(device)
        # heatmaps = heatmaps.to(device)
        # vismaps = vismaps.to(device)
        # non_blocking是针对GPU上的内存（显存），表示把数据锁页在显存上，在后台进程过程中不释放。
        # 一般地，如果pin_momery为True，把non_blocking也设为True，有助于加速数据传输，加快训练过程
        data = data.cuda(non_blocking=True)
        heatmaps = heatmaps.cuda(non_blocking=True)
        vismaps = vismaps.cuda(non_blocking=True)
        heat_pred1, heat_pred2 = net(data)
        print("heat_pred1:", len(heat_pred1))
        # 计算损失函数（总损失+L1+L2）
        loss_output = loss(heatmaps, heat_pred1, heat_pred2, vismaps)
        print('loss_output:', loss_output)
        # 将模型的参数梯度初始化为0
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss_output[0].backward()
        # 更新所有参数
        optimizer.step()
        metrics.append([loss_output[0].item(), loss_output[1].item(), loss_output[2].item()])
        print('time using: %.6f' % (time.time() - s))
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time


def validate(data_loader, net, loss):
    start_time = time.time()
    net.eval()
    metrics = []
    for i, (data, heatmaps, vismaps) in enumerate(data_loader):
        data = data.cuda(non_blocking=True)
        heatmaps = heatmaps.cuda(non_blocking=True)
        vismaps = vismaps.cuda(non_blocking=True)
        heat_pred1, heat_pred2 = net(data)
        loss_output = loss(heatmaps, heat_pred1, heat_pred2, vismaps)
        metrics.append([loss_output[0].item(), loss_output[1].item(), loss_output[2].item()])
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time


if __name__ == '__main__':
    # 创建一个 ArgumentParser 对象。
    # ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--clothes', help='specify the clothing type', default='outwear')
    parser.add_argument('-r', '--resume', help='specify the checkpoint', default=None)
    args = parser.parse_args(sys.argv[1:])
    print('Training ' + args.clothes)

    config = Config(args.clothes)
    # config = Config("blouse")
    workers = config.workers
    n_gpu = pytorch_utils.setgpu(config.gpus)
    # 大的batchsize减少训练时间，提高稳定性
    # 也会导致模型泛化能力下降
    # Hoffer[7]等人的研究表明，大的batchsize性能下降是因为训练时间不够长，本质上并不少batchsize的问题，
    # 在同样的epochs下的参数更新变少了，因此需要更长的迭代次数。
    batch_size = config.batch_size_per_gpu * n_gpu

    epochs = config.epochs
    # 256 pixels: SGD L1 loss starts from 1e-2, L2 loss starts from 1e-3
    # 512 pixels: SGD L1 loss starts from 1e-3, L2 loss starts from 1e-4
    base_lr = config.base_lr
    save_dir = config.proj_path + 'checkpoints/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 生成一个CPN网络实例，CPN（级联金字塔网络）=GlobalNet+RefineNet
    net = CascadePyramidNet(config)
    # 生成一个损失函数实例
    loss = VisErrorLoss()
    # 生成训练集和验证集的数据加载器
    train_data = KPDA(config, config.data_path, 'train')
    val_data = KPDA(config, config.data_path, 'val')
    print('Train sample number: %d' % train_data.size())
    print('Val sample number: %d' % val_data.size())

    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'

    # 将模型加载到GPU上
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # print('Using device:', device)
    # net = net.to(device)
    # loss = loss.to(device)
    net = net.cuda()
    loss = loss.cuda()
    # 大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    cudnn.benchmark = True
    # 在多卡的GPU服务器，当我们在上面跑程序的时候，当迭代次数或者epoch足够大的时候，我们通常会使用nn.DataParallel函数来用多个GPU来加速训练
    net = DataParallel(net)

    train_dataset = DataGenerator(config, train_data, phase='train')
    # print(len(train_dataset))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    val_dataset = DataGenerator(config, val_data, phase='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            collate_fn=val_dataset.collate_fn,
                            pin_memory=True)
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    lrs = LRScheduler(lr, patience=3, factor=0.1, min_lr=0.01 * lr, best_loss=best_val_loss)
    print('Start training from epoch %d...' % start_epoch)
    for epoch in range(start_epoch, epochs + 1):
        print('Epoch %d, lr = %.6f' % (epoch, lr))
        train_metrics, train_time = train(train_loader, net, loss, optimizer, lr)
        print('train_time: %.2f' % train_time)
        with torch.no_grad():
            val_metrics, val_time = validate(val_loader, net, loss)
        print('val_time: %.2f' % val_time)

        # 记录训练日志
        print_log(epoch, lr, train_metrics, train_time, val_metrics, val_time, save_dir=save_dir, log_mode=log_mode)

        val_loss = np.mean(val_metrics[:, 0])
        lr = lrs.update_by_rule(val_loss)
        if val_loss < best_val_loss or epoch % 10 == 0 or lr is None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict': state_dict,
                    'lr': lr,
                    'best_val_loss': best_val_loss},
                    os.path.join(save_dir, 'kpt_' + config.clothes + '_%03d.ckpt' % epoch))

        if lr is None:
            print('Training is early-stopped')
            break
