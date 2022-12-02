import comet_ml
import hashlib
import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter
from torchmetrics.functional import precision_recall, accuracy
from torchmetrics.functional import auc
from torchmetrics.functional import f1_score
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanAbsolutePercentageError

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
    


def get_experiment(project_name, run_id):
    experiment_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
    os.environ["COMET_EXPERIMENT_KEY"] = experiment_id

    api = comet_ml.API('3YfcpxE1bYPCpkkg4pQ2OjQ2r')  # Assumes API key is set in config/env
    api_experiment = api.get_experiment_by_key(experiment_id)

    if api_experiment is None:
        return comet_ml.Experiment(api_key ='3YfcpxE1bYPCpkkg4pQ2OjQ2r', project_name = project_name)

    else:
        return comet_ml.ExistingExperiment(api_key ='3YfcpxE1bYPCpkkg4pQ2OjQ2r', project_name = project_name)

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true', help='Fused window shift & window partition, similar for reversed part.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config




def main(config):
    local_rank = int(os.environ["LOCAL_RANK"])
    
    experiment = get_experiment(config.MODEL.PROJECT_NAME, config.MODEL.RUN_ID)
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters}")
    
    if config.PARALLEL_TYPE == 'ddp':
        model.cuda()
    
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    print('Optimizer built!')
    print(f'Local rank from environment: {local_rank}')
    print(f'Local rank from config: {config.LOCAL_RANK}')
    if config.PARALLEL_TYPE == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        print('Model wrapped in DDP!')
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
        
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_reg = torch.nn.L1Loss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model, experiment)
        if config.MODEL.TASK_TYPE == 'cls':
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            experiment.log_metric('test_accuracy', acc1)
            experiment.log_metric('test_f1_score', f1_score)
            experiment.log_metric('test_recall', recall)
            experiment.log_metric('test_precision', precision)
            experiment.log_metric('test_loss_cls', loss_cls)
        elif config.MODEL.TASK_TYPE == 'reg':
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_reg:.4f}")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model, experiment)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
        logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
        logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
        experiment.log_metric('test_accuracy', acc1)
        experiment.log_metric('test_f1_score', f1_score)
        experiment.log_metric('test_recall', recall)
        experiment.log_metric('test_precision', precision)
        experiment.log_metric('test_loss_cls', loss_cls)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch) if config.PARALLEL_TYPE == 'ddp' else None

        train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, experiment)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model, experiment)
        if config.MODEL.TASK_TYPE == 'cls':
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            experiment.log_metric('test_accuracy', acc1)
            experiment.log_metric('test_f1_score', f1_score)
            experiment.log_metric('test_recall', recall)
            experiment.log_metric('test_precision', precision)
            experiment.log_metric('test_loss_cls', loss_cls)
        elif config.MODEL.TASK_TYPE == 'reg':
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_reg:.4f}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))