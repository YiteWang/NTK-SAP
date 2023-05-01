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

    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '23456'

    ## Data ##
    args.input_shape, args.num_classes = load.dimension(args.dataset) 

    # Start to DDP
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    global best_test_acc1
    global best_test_epoch
    print("Use GPU: {} for training".format(gpu))
    print(args.dist_url)
    args.rank = args.rank * args.ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                    world_size=args.world_size, rank=args.rank)

    # Create model for each node
    model = load.model(args.model, args.model_class)(input_shape=args.input_shape, 
                                             num_classes=args.num_classes, 
                                             dense_classifier=args.dense_classifier,
                                             pretrained=args.pretrained)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    args.train_batch_size = int(args.train_batch_size/args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    print("GPU: {}, batchsize:{}, workers:{}".format(gpu, args.train_batch_size, args.workers))
    
    loss = nn.CrossEntropyLoss().cuda(gpu)
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    if args.resume:
        assert os.path.isfile(args.resume), 'Unable to find checkpoint file.'
        print('Loading checkpoint : {}'.format(args.resume))
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.saved_best_test_acc1 = checkpoint['best_test_acc1']
        args.saved_best_test_epoch = checkpoint['best_test_epoch']
        print('Load checkpoint from {}, start from epoch {}'.format(args.resume, checkpoint['epoch']))
        # Check Sparsity
        total_params = 0
        remained_params = 0
        for m,p in generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual):
            total_params += m.nelement()
            remained_params += m.detach().cpu().sum()
        print('Mask with density {} found and loaded.'.format(1.0*remained_params/total_params))
    else:
        print('[*] No checkpoint found, start from Dense and Normal.')

    torch.backends.cudnn.benchmark = True

    train_loader, sampler = load.dataloader(args.dataset, args.train_batch_size, True, args.workers, ddp=True)
    test_loader, _ = load.dataloader(args.dataset, args.test_batch_size, False, args.workers, ddp=True)

    device = 'cuda:{}'.format(gpu)
    # Train Model
    train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                  test_loader, device, args.post_epochs, args.verbose, sampler=sampler, gpu=gpu, args=args)
    

