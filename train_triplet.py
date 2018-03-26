from __future__ import print_function

import argparse
import os
import shutil
import pprint
import time
import random
import datetime
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.vehicle_reid as customized_models
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from torch.autograd import Variable

import numpy as np
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


try:
    import accimage
except ImportError:
    accimage = None
import numpy as np

from tools import draw_curve
from config import cfg, cfg_from_file, cfg_from_list
from utils import Logger, AverageMeter, measure_model, accuracy, mkdir_p, savefig, weight_filler, RandomPixelJitter
root='/home/priv-lab1/Database/vehicleid/'
# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))
customized_models_names = sorted(name for name in customized_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))
for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]
model_names = default_model_names + customized_models_names
print(model_names)

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/cls/resnet18_1x64d_imagenet.yml', type=str)
parser.add_argument('--set', dest='set_cfgs',
                    help='set config keys', default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('--num-instances', type=int, default=4,
                    help="each minibatch consist of "
                         "(batch_size // num_instances) identities, and "
                         "each identity has num_instances instances, "
                         "default: 4")
parser.add_argument('--margin', type=float, default=0.5,
                    help="margin of the triplet loss, default: 0.5")
# parser.add_argument('--pretrained', default='/home/priv-lab1/workspace/pytorch-priv/model_best.pth.tar',
#                     help="pretrain model")
args = parser.parse_args()
print('==> Called with args:')
print(args)
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
print('==> Using config:')
pprint.pprint(cfg)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_ids
USE_CUDA = torch.cuda.is_available()
# Random seed
if cfg.rng_seed is None:
    cfg.rng_seed = random.randint(1, 10000)
random.seed(cfg.rng_seed)
torch.manual_seed(cfg.rng_seed)
if USE_CUDA:
    torch.cuda.manual_seed_all(cfg.rng_seed)
# Global param
BEST_ACC = 0  # best test accuracy
LR_STATE = cfg.CLS.base_lr

class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        # dist_ap_max = torch.max(dist_ap)
        # dist_an_min = torch.min(dist_an)
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)
def pil_resize(im, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = im.size
        if (w <= h and w == size) or (h <= w and h == size):
            return im
        if w < h:
            ow = size
            oh = int(size * h / w)
            return im.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return im.resize((ow, oh), interpolation)
    else:
        return im.resize(size[::-1], interpolation)
def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    global BEST_ACC, LR_STATE
    # switch to train mode
    if not cfg.CLS.fix_bn:
        model.train()
    else:
        model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # adjust learning rate
        adjust_learning_rate(optimizer, epoch, batch=batch_idx, batch_per_epoch=len(train_loader))

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # forward pass: compute output
        outputs = model(inputs)
        # forward pass: compute gradient and do SGD step
        optimizer.zero_grad()
        loss,prec1 = criterion(outputs, targets)
        # backward
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        #prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1, inputs.size(0))
        #top1.update(prec1[0], inputs.size(0))
        #top5.update(prec5[0], inputs.size(0))

        if (batch_idx + 1) % cfg.CLS.disp_iter == 0:
            print('Training: [{}/{}][{}/{}] | Time: {:.2f} | Data: {:.2f} | '
                  'LR: {:.8f} | Top1: {:.2%} | Loss: {:.4f} | Total: {:.2f}'
                  .format(epoch + 1, cfg.CLS.epochs, batch_idx + 1, len(train_loader), batch_time.average(),
                          data_time.average(), LR_STATE, top1.avg, losses.avg,
                          batch_time.sum + data_time.sum))

    return (losses.avg, top1.avg)


def test(val_loader, model, criterion, epoch, use_cuda):
    global BEST_ACC, LR_STATE

    print('==> Evaluating at {} epochs...'.format(epoch + 1))
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # forward pass: compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        print('Testing: [{}/{}][{}/{}] | Best_Acc: {:4.2f}% | Time: {:.2f} | Data: {:.2f} | '
              'LR: {:.8f} | Top1: {:.4f}% | Top5: {:.4f}% | Loss: {:.4f} | Total: {:.2f}'
              .format(epoch + 1, cfg.CLS.epochs, batch_idx + 1, len(val_loader), BEST_ACC,
                      batch_time.average(), data_time.average(),
                      LR_STATE, top1.avg, top5.avg, losses.avg, batch_time.sum + data_time.sum))

    return (losses.avg, top1.avg, top5.avg)


def save_checkpoint(model, optimizer, cur_acc, epoch):
    global BEST_ACC
    print('==> Saving checkpoints...')
    suffix_latest = 'latest.pth.tar'
    suffix_best = 'best.pth.tar'

    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'acc': cur_acc, 'best_acc': BEST_ACC,
             'optimizer': optimizer.state_dict()}

    torch.save(state, '{}/model_{}'.format(cfg.CLS.ckpt, suffix_latest))

    if cur_acc > BEST_ACC:
        # update BEST_ACC
        BEST_ACC = cur_acc
        shutil.copyfile('{}/model_{}'.format(cfg.CLS.ckpt, suffix_latest),
                        '{}/model_{}'.format(cfg.CLS.ckpt, suffix_best))


def adjust_learning_rate(optimizer, epoch, batch=0, batch_per_epoch=5000):
    global LR_STATE
    if cfg.CLS.cosine_lr:
        total_batch = cfg.CLS.epochs * batch_per_epoch
        cur_batch = (epoch % cfg.CLS.epochs) * batch_per_epoch + batch
        LR_STATE = 0.5 * cfg.CLS.base_lr * (np.cos(cur_batch * np.pi / total_batch) + 1.0)
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
            if cur_lr / LR_STATE > 10 - cfg.eps5:
                param_group['lr'] = LR_STATE * 10
            else:
                param_group['lr'] = LR_STATE
    else:
        if epoch in cfg.CLS.schedule and batch == 0:
            LR_STATE *= cfg.CLS.gamma
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                if cur_lr / LR_STATE > (1.0 / cfg.CLS.gamma) + cfg.eps5:
                    param_group['lr'] = LR_STATE * 10
                else:
                    param_group['lr'] = LR_STATE
    pass
def default_loader(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((256,256))
    img = RandomErasing(img, 0.5)
    return img

class CustomData(Dataset):
    def __init__(self, img_path, txt_path, data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[1]) for line in lines]
        self.data_transforms = data_transforms
        self.img_path=img_path
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]

        img = self.loader(root + img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)

            except:
                print("Cannot transform image: {}".format(img_name))
        return img,label

def main():
    global BEST_ACC, LR_STATE
    start_epoch = cfg.CLS.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Create ckpt folder
    if not os.path.isdir(cfg.CLS.ckpt):
        mkdir_p(cfg.CLS.ckpt)
    if args.cfg_file is not None and not cfg.CLS.evaluate:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.CLS.ckpt, args.cfg_file.split('/')[-1]))

    # Dataset and Loader
    normalize = transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
    if cfg.CLS.train_crop_type == 'center':
        train_aug = [
                     transforms.Resize(cfg.CLS.base_size),
                     transforms.CenterCrop(cfg.CLS.crop_size),
                     transforms.RandomHorizontalFlip(),
                    ]
    elif cfg.CLS.train_crop_type == 'random_resized':
        train_aug = [transforms.RandomResizedCrop(cfg.CLS.crop_size),
                     transforms.RandomHorizontalFlip()]
    else:
        train_aug = [transforms.RandomHorizontalFlip()]
    if len(cfg.CLS.rotation) > 0:
        train_aug.append(transforms.RandomRotation(cfg.CLS.rotation))
    if len(cfg.CLS.pixel_jitter) > 0:
        train_aug.append(RandomPixelJitter(cfg.CLS.pixel_jitter))
    if cfg.CLS.grayscale > 0:
        train_aug.append(transforms.RandomGrayscale(cfg.CLS.grayscale))
    train_aug.append(transforms.ToTensor())
    train_aug.append(normalize)

    val_aug = [
                transforms.Resize(cfg.CLS.base_size),
               transforms.CenterCrop(cfg.CLS.crop_size),
               transforms.ToTensor(),
               normalize, ]
    if os.path.isfile(cfg.CLS.train_root):
        # if cfg.CLS.have_data_list:
        train_datasets = CustomData(img_path=cfg.CLS.data_root,
                                    txt_path=cfg.CLS.train_root,
                                    data_transforms=transforms.Compose(train_aug))

        val_datasets = CustomData(img_path=cfg.CLS.data_root,
                                  txt_path=cfg.CLS.val_root,
                                  data_transforms=transforms.Compose(val_aug))
        # else:
    elif os.path.isdir(cfg.CLS.data_root + cfg.CLS.train_root):
        traindir = os.path.join(cfg.CLS.data_root, cfg.CLS.train_root)
        train_datasets = datasets.ImageFolder(traindir, transforms.Compose(train_aug))

        valdir = os.path.join(cfg.CLS.data_root, cfg.CLS.val_root)
        val_datasets = datasets.ImageFolder(valdir, transforms.Compose(val_aug))

    train_loader = torch.utils.data.DataLoader(train_datasets,
                                               batch_size=cfg.CLS.train_batch, shuffle=False,
                                               sampler=RandomIdentitySampler(train_datasets,num_instances=4),
                                               num_workers=cfg.workers, pin_memory=True,drop_last=True)
    print(type(train_loader))

    if cfg.CLS.validate or cfg.CLS.evaluate:
        val_loader = torch.utils.data.DataLoader(val_datasets,
                                                 batch_size=cfg.CLS.test_batch, shuffle=False,
                                                 num_workers=cfg.workers, pin_memory=True,drop_last=True)

    # Create model
    model = models.__dict__[cfg.CLS.arch]()
    print(model)
    # Calculate FLOPs & Param
    n_flops, n_convops, n_params = measure_model(model, cfg.CLS.crop_size, cfg.CLS.crop_size)
    print('==> FLOPs: {:.4f}M, Conv_FLOPs: {:.4f}M, Params: {:.4f}M'.
          format(n_flops / 1e6, n_convops / 1e6, n_params / 1e6))
    del model
    model = models.__dict__[cfg.CLS.arch]()

    # Load pre-train model
    if cfg.CLS.pretrained:
        print("==> Using pre-trained model '{}'".format(cfg.CLS.pretrained))
        pretrained_dict = torch.load(cfg.CLS.pretrained)
        try:
            pretrained_dict = pretrained_dict['state_dict']
        except:
            pretrained_dict = pretrained_dict
        model_dict = model.state_dict()
        updated_dict, match_layers, mismatch_layers = weight_filler(pretrained_dict, model_dict)
        model_dict.update(updated_dict)
        model.load_state_dict(model_dict)
    else:
        print("==> Creating model '{}'".format(cfg.CLS.arch))

    # Define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = TripletLoss(margin=args.margin).cuda()
    if cfg.CLS.pretrained:
        def param_filter(param):
            return param[1]

        new_params = map(param_filter, filter(lambda p: p[0] in mismatch_layers, model.named_parameters()))
        base_params = map(param_filter, filter(lambda p: p[0] in match_layers, model.named_parameters()))
        model_params = [{'params': base_params}, {'params': new_params, 'lr': cfg.CLS.base_lr * 10}]
    else:
        model_params = model.parameters()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    optimizer = optim.SGD(model_params, lr=cfg.CLS.base_lr, momentum=cfg.CLS.momentum,
                          weight_decay=cfg.CLS.weight_decay)

    # Evaluate model
    if cfg.CLS.evaluate:
        print('\n==> Evaluation only')
        test_loss, test_top1, test_top5 = test(val_loader, model, criterion, start_epoch, USE_CUDA)
        print('==> Test Loss: {:.8f} | Test_top1: {:.4f}% | Test_top5: {:.4f}%'.format(test_loss, test_top1, test_top5))
        return

    # Resume training
    title = 'Pytorch-CLS-' + cfg.CLS.arch
    if cfg.CLS.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint '{}'".format(cfg.CLS.resume))
        assert os.path.isfile(cfg.CLS.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(cfg.CLS.resume)
        BEST_ACC = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(cfg.CLS.ckpt, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(cfg.CLS.ckpt, 'log.txt'), title=title)
        #logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc.'])

    # Train and val
    for epoch in range(start_epoch, cfg.CLS.epochs):
        print('\nEpoch: [{}/{}] | LR: {:.8f}'.format(epoch + 1, cfg.CLS.epochs, LR_STATE))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, USE_CUDA)
        # top1 =  train_acc
        # BEST_ACC = max(top1, BEST_ACC)
        if cfg.CLS.validate:
            #test_loss, test_top1, test_top5 = test(val_loader, model, criterion, epoch, USE_CUDA)
            top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)
            best_acc = max(top1, BEST_ACC)

            print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, top1, best_acc))
        #else:
            #test_loss, test_top1, test_top5 = 0.0, 0.0, 0.0

        # Append logger file
        #logger.append([LR_STATE, train_loss, test_loss, train_acc, test_top1])
        logger.append([LR_STATE, train_loss , train_acc])

        # Save model
        save_checkpoint(model, optimizer, train_acc, epoch)
        # Draw curve
        try:
            draw_curve(cfg.CLS.arch, cfg.CLS.ckpt)
            print('==> Success saving log curve...')
        except:
            print('==> Saving log curve error...')

    logger.close()
    try:
        savefig(os.path.join(cfg.CLS.ckpt, 'log.eps'))
        shutil.copyfile(os.path.join(cfg.CLS.ckpt, 'log.txt'), os.path.join(cfg.CLS.ckpt, 'log{}.txt'.format(
            datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
    except:
        print('Copy log error.')
    print('==> Training Done!')
    print('==> Best acc: {:.4f}%'.format(best_top1))
def RandomErasing(img_ ,probability):

    if random.uniform(0, 1) > probability:
        return img_

    for attempt in range(100):
        area = img_.size[0] * img_.size[1]

        target_area = random.uniform(0.02, 0.4) * area
        aspect_ratio = random.uniform(0.3, 1 / 0.3)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img_.size[1] and h < img_.size[0]:

            x1 = random.randint(0, img_.size[0] - h)
            y1 = random.randint(0, img_.size[1] - w)

            img_ = np.array(img_)
            img_[x1:x1 + h, y1:y1 + w,0] = 0.4914 * 255
            img_[x1:x1 + h, y1:y1 + w,1] = 0.4822 * 255
            img_[x1:x1 + h, y1:y1 + w,2] = 0.4465 * 255
            img_ = Image.fromarray(img_)



            return img_

    return img_

if __name__ == '__main__':
    main()
