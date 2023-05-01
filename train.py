import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from Utils import generator

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10, gpu=None):
    model.train()
    total = 0
    correct1 = 0
    correct5 = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(dataloader):
        if gpu:
            data, target = data.cuda(gpu, non_blocking=True), target.cuda(gpu, non_blocking=True)
        else:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        _, pred = output.topk(5, dim=1)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        correct1 += correct[:,:1].sum().item()
        correct5 += correct[:,:5].sum().item()
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    end_time = time.time()
    print('Epoch: {}, Train: Top 1 Accuracy: {}/{} ({:.2f}%), Time Used: {} mins'.format(
             epoch, correct1, len(dataloader.dataset), accuracy1, (end_time-start_time)/60.0))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose, gpu=None):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            if gpu:
                data, target = data.cuda(gpu, non_blocking=True), target.cuda(gpu, non_blocking=True)
            else:
                data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    # if verbose:
    print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, sampler=None, gpu=None, args=None):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    if args is not None:
        best_test_epoch = args.saved_best_test_epoch
        best_test_acc1 = args.saved_best_test_acc1
        start_epoch = args.start_epoch
    else:
        best_test_epoch = 0
        best_test_acc1 = 0
        start_epoch = 0

    print('Start from {} with best epoch/acc: {}, {}'.format(start_epoch, best_test_epoch, best_test_acc1))

    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose, gpu=gpu)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose, gpu=gpu)
        if accuracy1 > best_test_acc1:
            best_test_acc1 = accuracy1
            best_test_epoch = epoch
        row = [train_loss, test_loss, accuracy1, accuracy5]
        rows.append(row)
        scheduler.step()
        if args is not None:
            if args.rank % args.ngpus_per_node == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_test_acc1': best_test_acc1,
                    'best_test_epoch': best_test_epoch,
                }, "{}/training_model.pth".format(args.result_dir))

                # Save as best checkpoint
                if best_test_epoch == epoch:
                    torch.save({
                    'epoch': epoch + 1,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_test_acc1': best_test_acc1,
                    'best_test_epoch': best_test_epoch,
                }, "{}/best_model.pth".format(args.result_dir))


    print('Best test accuracy is {} at epoch {}'.format(best_test_acc1, best_test_epoch))
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)


