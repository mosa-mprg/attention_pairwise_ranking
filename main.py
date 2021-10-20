import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import pickle
import cv2
import numpy as np
from dataset import TSNDataSet
from resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from transforms import *
from opts import parser
from os import path
from ops.basic_ops import ConsensusModule

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

best_prec = 0
def main():
    global args, best_prec
    start_epoch = args.start_epoch
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    use_cuda = torch.cuda.is_available()

    if  args.dataset == 'pairwise':
        num_class = 1
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    # Data loading code
    normalize = IdentityTransform()

    data_length = 1 #num frame
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   image_tmpl="{:06d}.jpg",
                   transform=torchvision.transforms.Compose([
                       GroupMultiScaleCrop(224, [1, .875, .75, .66]),GroupRandomHorizontalFlip(is_flow=False),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   image_tmpl="{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(256),
                       GroupCenterCrop(224),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = torch.nn.MarginRankingLoss(margin=1.0).cuda()

    if  args.arch == 'resnet18':
        model = resnet18()
    elif  args.arch == 'resnet34':
        model = resnet34()
    elif  args.arch == 'resnet50':
        model = resnet50()
    elif  args.arch == 'resnet101':
        model = resnet101()
    else:
        print("{} is nan".format(args.arch))

    if args.resume:
        if os.path.isfile(args.resume):
            print(("{}".format(args.resume)))
            checkpoint = torch.load(args.resume)
            best_prec = checkpoint['best_prec']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(model.parameters(), args.lr)

    if args.evaluate:
        with torch.no_grad():
            validate(val_loader, model, 0 , use_cuda, data_length, criterion)
        return
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch, use_cuda, data_length, criterion)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                prec = validate(val_loader, model, (epoch + 1) * len(train_loader), use_cuda, data_length, criterion)
                is_best = prec > best_prec
                best_prec = max(prec, best_prec)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_prec': best_prec,
                }, is_best)

def train(train_loader, model, optimizer, epoch, use_cuda, data_length, criterion):

    av_meters = {'batch_time': AverageMeter(), 'data_time': AverageMeter(),
                 'losses': AverageMeter(),'acc_good': AverageMeter(),'acc_bad': AverageMeter()}

    model.train()
    end = time.time()

    for i, (input1, input2) in enumerate(train_loader):
        av_meters['data_time'].update(time.time() - end)
        if use_cuda:
            input1,input2, labels1, labels2 = input1.cuda(),input2.cuda(), torch.ones(input1.size(0)).cuda(), torch.ones(input2.size(0)).cuda()
        input1_var = torch.autograd.Variable(input1, requires_grad=True)
        input2_var = torch.autograd.Variable(input2, requires_grad=True)
        target1  = torch.autograd.Variable(labels1, requires_grad=False)
        target2  = torch.autograd.Variable(labels2, requires_grad=False)
        input1_var = input1_var.view((-1, 3) + input1_var.size()[-2:])
        input2_var = input2_var.view((-1, 3) + input2_var.size()[-2:])

        att_output1_good, att_output1_bad, output1_good, output1_bad, _ = model(input1_var)
        att_output2_good, att_output2_bad, output2_good, output2_bad, _ = model(input2_var)
        ranking_good_loss, prec_good = acc_los(criterion, output1_good, output2_good, target1)
        ranking_bad_loss, prec_bad = acc_los(criterion, output2_bad, output1_bad, target2)
        att_good_loss, _ = acc_los(criterion, att_output1_good, att_output2_good, target1)
        att_bad_loss, _ = acc_los(criterion, att_output2_bad, att_output1_bad, target2)
        loss =   att_good_loss + att_bad_loss + ranking_good_loss + ranking_bad_loss

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        av_meters['losses'].update(loss.item(), input1.size(0))
        av_meters['acc_good'].update(prec_good, input1.size(0))
        av_meters['acc_bad'].update(prec_bad, input1.size(0))
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            console_log_train(av_meters, epoch, i, len(train_loader), )

def validate(val_loader, model, iter, use_cuda, data_length, criterion, logger=None):
    av_meters = {'batch_time': AverageMeter(), 'losses': AverageMeter(),
                 'acc_good': AverageMeter(),'acc_bad': AverageMeter(),
                 'ranking_good_loss': AverageMeter()}
    # switch to evaluate mode
    model.eval()
    end = time.time()
    count = 0
    for i, (input1, input2) in enumerate(val_loader):

        if use_cuda:
            input1, input2, labels1, labels2 = input1.cuda(),input2.cuda(), torch.ones(input1.size(0)).cuda(), torch.ones(input2.size(0)).cuda()
        input1_var = torch.autograd.Variable(input1, requires_grad=True)
        input2_var = torch.autograd.Variable(input2, requires_grad=True)
        target1  = torch.autograd.Variable(labels1, requires_grad=False)
        target2  = torch.autograd.Variable(labels2, requires_grad=False)
        input1_var = input1_var.view((-1, 3) + input1_var.size()[-2:])
        input2_var = input2_var.view((-1, 3) + input2_var.size()[-2:])

        _, _, output1_good, output1_bad, attention1 = model(input1_var)
        _, _, output2_good, output2_bad, attention2 = model(input2_var)
        ranking_good_loss, prec_good = acc_los(criterion, output1_good, output2_good, target1)
        ranking_bad_loss, prec_bad = acc_los(criterion, output2_bad, output1_bad, target2)
        loss = ranking_good_loss + ranking_bad_loss

        
        if args.evaluate:
            att_output_path = path.join('/mount_data/APR_demo/image', '{}'.format(args.board))
            for j in range(2):
                if j == 0:
                    attention_good,attention_bad,fe, per_good, per_bad = attention1
                    inputs = input1_var
                    cnn = 1
                else:
                    attention_good,attention_bad, fe, per_good, per_bad = attention2
                    inputs = input2_var
                    cnn = 2

                c_att_good = attention_good.data.cpu()
                c_att_good = c_att_good.numpy()
                c_att_bad = attention_bad.data.cpu()
                c_att_bad = c_att_bad.numpy()
                d_inputs = inputs.data.cpu()
                d_inputs = d_inputs.numpy()
                d_inputs = d_inputs.reshape(-1,3,224,224)
                in_b, in_c, in_y, in_x = inputs.shape
                for count, (item_img, item_att_good,item_att_bad) in enumerate(zip(d_inputs, c_att_good,c_att_bad)):
                    v_img = ((item_img.transpose((1,2,0))*[0.5, 0.5, 0.5])+ [0.5, 0.5, 0.5])* 255
                    v_img = v_img[:, :, ::-1]
                    resize_att_good = cv2.resize(item_att_good[0], (in_x, in_y))
                    resize_att_good *= 255.
                    jet_map0_good = cv2.applyColorMap(resize_att_good.astype(np.uint8), cv2.COLORMAP_JET)
                    jet_map_good = cv2.addWeighted(jet_map0_good,0.3,v_img.astype(np.uint8),0.7,0)
                    resize_att_bad = cv2.resize(item_att_bad[0], (in_x, in_y))
                    resize_att_bad *= 255.
                    jet_map0_bad = cv2.applyColorMap(resize_att_bad.astype(np.uint8), cv2.COLORMAP_JET)
                    jet_map_bad = cv2.addWeighted(jet_map0_bad,0.3,v_img.astype(np.uint8),0.7,0)

                    out_path = path.join(att_output_path, 'good_blend', '{0}{1:06d}.png'.format(cnn,count))
                    cv2.imwrite(out_path, jet_map_good)
                    out_path = path.join(att_output_path, 'good_attention', '{0}{1:06d}.png'.format(cnn,count))
                    cv2.imwrite(out_path, jet_map0_good)
                    out_path = path.join(att_output_path, 'bad_blend', '{0}{1:06d}.png'.format(cnn,count))
                    cv2.imwrite(out_path, jet_map_bad)
                    out_path = path.join(att_output_path, 'bad_attention', '{0}{1:06d}.png'.format(cnn,count))
                    cv2.imwrite(out_path, jet_map0_bad)
                    out_path = path.join(att_output_path, 'raw', '{0}{1:06d}.png'.format(cnn,count))
                    cv2.imwrite(out_path, v_img)
                    count += 1
        

        av_meters['losses'].update(loss.item(), input1.size(0))
        av_meters['acc_good'].update(prec_good, input1.size(0))
        av_meters['acc_bad'].update(prec_bad, input1.size(0))
        av_meters['batch_time'].update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            console_log_test(av_meters, i, len(val_loader))
    print(('Testing Results: Acc_good {acc_good.avg:.3f} Acc_bad {acc_bad.avg:.3f}'
           .format(acc_good=av_meters['acc_good'],acc_bad=av_meters['acc_bad'])))
    return av_meters['acc_good'].avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, 'best_model.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.lr_steps:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def acc_los(criterion, input1, input2, target):

    output1 = input1.view((-1, args.num_segments) + input1.size()[1:])
    output1 = output1.mean(dim=1, keepdim=True)
    output1 = output1.squeeze(1)

    output2 = input2.view((-1, args.num_segments) + input2.size()[1:])
    output2 = output2.mean(dim=1, keepdim=True)
    output2 = output2.squeeze(1)

    pred1 = output1.data
    pred2 = output2.data
    correct = torch.gt(pred1, pred2)
    acc = float(correct.sum())/correct.size(0)

    #loss = torch.mean(torch.log(torch.exp(output2 - output1)+1))
    loss = criterion(output1, output2, target)

    return loss, acc

def console_log_train(av_meters, epoch, iter, epoch_len):
    print(('Epoch: [{0}][{1}/{2}]\t'
           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
           'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
           'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
           'Prec_good: {acc_good.val:.3f} ({acc_good.avg:.3f})\t'
           'Prec_bad: {acc_bad.val:.3f} ({acc_bad.avg:.3f})'.format(
               epoch, iter, epoch_len, batch_time=av_meters['batch_time'],
               data_time=av_meters['data_time'], loss=av_meters['losses'],
               acc_good=av_meters['acc_good'], acc_bad=av_meters['acc_bad'])))

def console_log_test(av_meters, iter, test_len):
    print(('Test: [{0}/{1}\t'
           'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
           'Prec_good {acc_good.val} ({acc_good.avg:.3f})\t'
           'Prec_bad {acc_bad.val} ({acc_bad.avg:.3f})'.format(
               iter, test_len, batch_time=av_meters['batch_time'], loss=av_meters['losses'],
               acc_good=av_meters['acc_good'], acc_bad=av_meters['acc_bad'])))

if __name__ == '__main__':
    main()
