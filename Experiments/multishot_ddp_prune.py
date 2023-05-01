import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(args):
    if not args.save:
        print("This experiment requires an expid.")
        quit()

    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)
    multi_gpu = torch.cuda.device_count() > 1 and args.gpu == 'all'

    ## We prune the neural networks with nn.dataparallel

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    args.input_shape, args.num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * args.num_classes)

    ## Model ##
    print('Creating {} model.'.format(args.model))
    model = load.model(args.model, args.model_class)(input_shape=args.input_shape, 
                                                     num_classes=args.num_classes, 
                                                     dense_classifier=args.dense_classifier,
                                                     pretrained=args.pretrained)

    # Use dataparallel
    if multi_gpu:
        print('Use {} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Save Original ##
    if multi_gpu:
        chekpoint = {'epoch': 0,
                     'model': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'best_test_acc1': 0,
                     'best_test_epoch': 0,}
    else:
        chekpoint = {'epoch': 0,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'best_test_acc1': 0,
                     'best_test_epoch': 0,}

    torch.save(chekpoint,"{}/init_model.pth".format(args.result_dir))
    # del chekpoint


    ## Train-Prune Loop ##
    for compression in args.compression_list:
        print('{} compression ratio'.format(compression))

        if args.pruner == 'synflow' and args.model_class == 'imagenet':
            print('Using double version of Synflow')
            model.double()

        # Prune Model
        pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual), 
            ntksap_epsilon=args.ntksap_epsilon, ntksap_R=args.ntksap_R)

        # sparsity = (0.8**(float(compression)))**((l + 1) / level)
        sparsity = (0.8**(float(compression)))
        prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                   args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)

        total_params = 0
        remained_params = 0
        for m, p in pruner.masked_parameters:
            total_params += m.nelement()
            remained_params += m.detach().cpu().sum()
        print('Mask with sparsity {} found.'.format(1.0*remained_params/total_params))

        # Check the last layer
        last_m, last_p = pruner.masked_parameters[-1]
        if last_m.detach().cpu().sum()/last_m.nelement() == 0:
            print('Fully connected layer all pruned, expect not trainable.')

        if args.pruner == 'synflow' and args.model_class == 'imagenet':
            print('Convert model to float for saving checkpoint.')
            model.float()

        original_dict = torch.load("{}/init_model.pth".format(args.result_dir))
        # Save searched masks
        if multi_gpu:
            current_model_dict = model.module.state_dict()
            found_mask = dict(filter(lambda v: (v[0].endswith(('mask',))), current_model_dict.items()))
            original_dict['model'].update(found_mask)
        else:
            current_model_dict = model.state_dict()
            found_mask = dict(filter(lambda v: (v[0].endswith(('mask',))), current_model_dict.items()))
            original_dict['model'].update(found_mask)

        torch.save(original_dict,"{}/pruned_model_{}_{}.pth".format(args.result_dir, args.pruner, compression))
        # del chekpoint

        # prune_result = metrics.summary(model, 
        #                                pruner.scores,
        #                                metrics.flop(model, args.input_shape, device),
        #                                lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))





