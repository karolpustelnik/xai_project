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
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
from torchmetrics.functional import r2_score

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
        
    if config.DATA.PART == 'head':
        weights = torch.tensor([4.67852798, 0.43359265, 2.08357036]).cuda() # head class weights
    elif config.DATA.PART == 'femur':
        weights = torch.tensor([1.34257743, 0.76396232, 1.05686022]).cuda() # femur class weights
    elif config.DATA.PART == 'abdomen':
        weights = torch.tensor([4.51180538, 0.37685768, 8.01040497]).cuda() # abdomen class weights
    criterion_cls = torch.nn.CrossEntropyLoss() ## changed
    criterion_reg = torch.nn.L1Loss() ## changed

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
        if config.MODEL.TASK_TYPE == 'cls':
            print('Validating model after loading checkpoint...')
            acc1, f1_score, recall, precision, loss_meter_reg = validate(config, data_loader_val, model, experiment, epoch = 0)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"loss of the network on the {len(dataset_val)} test images: {loss_meter_reg}")
            max_accuracy = max(max_accuracy, acc1)
        elif config.MODEL.TASK_TYPE == 'reg':
            print('Validating model after loading checkpoint...')
            mae_meter, mape_meter, rmse_meter, loss_meter_reg = validate(config, data_loader_val, model, experiment, epoch = 0)
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_meter_reg:.4f}")
            logger.info(f"MAE of the network on the {len(dataset_val)} test images: {mae_meter:.4f}")
            logger.info(f"MAPE of the network on the {len(dataset_val)} test images: {mape_meter:.4f}")
            logger.info(f"RMSE of the network on the {len(dataset_val)} test images: {rmse_meter:.4f}")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if config.MODEL.TASK_TYPE == 'cls':
            print('Validating model after loading pretrained weights...')
            acc1, f1_score, recall, precision, loss_cls = validate(config, data_loader_val, model, experiment, epoch = 0)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
        elif config.MODEL.TASK_TYPE == 'reg':
            print('Validating model after loading pretrained weights...')
            mae_meter, mape_meter, rmse_meter, loss_meter_reg = validate(config, data_loader_val, model, experiment, epoch = 0)
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_meter_reg:.4f}")
            logger.info(f"MAE of the network on the {len(dataset_val)} test images: {mae_meter:.4f}")
            logger.info(f"MAPE of the network on the {len(dataset_val)} test images: {mape_meter:.4f}")
            logger.info(f"RMSE of the network on the {len(dataset_val)} test images: {rmse_meter:.4f}")

    logger.info("Start training")
    start_time = time.time()
    if config.EVAL_MODE == False:
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            data_loader_train.sampler.set_epoch(epoch) if config.PARALLEL_TYPE == 'ddp' else None

            train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, experiment)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)

            validate(config, data_loader_val, model, experiment, epoch)
            
    elif config.EVAL_MODE == True:
        acc1, f1_score, recall, precision,  loss_cls, loss_reg = validate(config, data_loader_val, model, experiment, epoch = 0)
        if config.MODEL.TASK_TYPE == 'cls':
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            logger.info(f"f1 score of the network on the {len(dataset_val)} test images: {f1_score*100}%")
            logger.info(f"recall of the network on the {len(dataset_val)} test images: {recall*100}%")
            logger.info(f"precision of the network on the {len(dataset_val)} test images: {precision*100}%")
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_cls:.4f}")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        elif config.MODEL.TASK_TYPE == 'reg':
            logger.info(f"Loss of the network on the {len(dataset_val)} test images: {loss_reg:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    
def train_one_epoch(config, model, criterion_cls, criterion_reg, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, experiment):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter() # stores average and current value
    loss_meter_cls = AverageMeter() 
    loss_meter_reg = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    for idx, (images, indexes, Class, video, measure, ps, frames_n, measure_normalized) in enumerate(data_loader): ## changed
        optimizer.zero_grad()
        
        if config.PARALLEL_TYPE == 'ddp':
            outputs = model(images)
            if config.MODEL.TYPE == 'resnet_50_attention':
                outputs = outputs[0]
        elif config.PARALLEL_TYPE == 'model_parallel':
            #labels = labels.to('cuda:1')
            outputs = model((images, frames_n))

        
        if config.MODEL.TASK_TYPE == 'cls':
            Class = Class.cuda(non_blocking=True)
            loss = criterion_cls(outputs, Class) ## changed
            
        elif config.MODEL.TASK_TYPE == 'reg':
            measure_normalized = measure_normalized.unsqueeze(0).cuda(non_blocking=True)
            measure_normalized = measure_normalized.reshape(-1, 1)
            loss = criterion_reg(outputs, measure_normalized)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update((epoch * num_steps + idx))

        torch.cuda.synchronize()
        if config.MODEL.TASK_TYPE == 'cls':
            loss_meter_cls.update(loss.item())
            
        if config.MODEL.TASK_TYPE == 'reg':
            loss_meter_reg.update(loss.item())

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'cls loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t'
                f'reg loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'lr {lr:.6f}\t')
        experiment.log_metric('learning rate', lr)
            #commet ml logging
            
    experiment.log_metric("lr", lr)
    if config.MODEL.TASK_TYPE == 'cls':
        experiment.log_metric("Train loss_cls", loss_meter_cls.avg, epoch=epoch)
    elif config.MODEL.TASK_TYPE == 'reg':
        experiment.log_metric("Train loss_reg", loss_meter_reg.avg, epoch = epoch)

            
            
            
@torch.no_grad()
def validate(config, data_loader, model, experiment, epoch):
    if config.DATA.PART == 'head':
        weights = torch.tensor([4.67852798, 0.43359265, 2.08357036]).cuda() # head class weights
    elif config.DATA.PART == 'femur':
        weights = torch.tensor([1.34257743, 0.76396232, 1.05686022]).cuda() # femur class weights
    elif config.DATA.PART == 'abdomen':
        weights = torch.tensor([4.51180538, 0.37685768, 8.01040497]).cuda() # abdomen class weights
    criterion_cls = torch.nn.CrossEntropyLoss() ## changed
    criterion_reg = torch.nn.L1Loss() ## changed
    mae = MeanAbsoluteError().cuda()
    mape = MeanAbsolutePercentageError().cuda()
    rmse = MeanSquaredError(squared = False).cuda()
    
    model.eval()
    
    loss_meter_cls = AverageMeter()
    loss_meter_reg = AverageMeter()
    acc1_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    f1_score_meter = AverageMeter()
    
    mae_meter = AverageMeter()
    mape_meter = AverageMeter()
    rmse_meter = AverageMeter()
    #r2_meter = AverageMeter()

    worst_losses = []
    for idx, (images, indexes, Class, video, measure, ps, frames_n, measure_normalized) in enumerate(data_loader):

        if config.PARALLEL_TYPE == 'ddp':
            #labels = labels.cuda(non_blocking=True)
            #print(labels.shape)
            outputs = model(images) 
            if config.MODEL.TYPE == 'resnet_50_attention':
                outputs = outputs[0]
            
        elif config.PARALLEL_TYPE == 'model_parallel':
            #labels = labels.to('cuda:1')
            outputs = model((images, frames_n))

        # measure accuracy and record loss
        if config.MODEL.TASK_TYPE == 'cls':
            Class = Class.cuda(non_blocking=True)
            print(f'shape of class: {Class.shape}')
            print(f'shape of outputs: {outputs.shape}')
            loss_cls = criterion_cls(outputs, Class)
            
        elif config.MODEL.TASK_TYPE == 'reg':
            measure_normalized = measure_normalized.unsqueeze(0).cuda(non_blocking=True)
            measure_normalized = measure_normalized.reshape(-1, 1)
            loss = criterion_reg(outputs, measure_normalized)
            if config.PARALLEL_TYPE == 'ddp':
                max_measure = 214.14944514917548 if Class[0] == 2 else 215.22020313394543 if Class[0] == 4 else 72.1250937626219
            elif config.PARALLEL_TYPE == 'model_parallel':
                max_measure = 214.14944514917548 # max head
            ps = ps.cuda(non_blocking=True)
            ps = ps.unsqueeze(1)
            predicted_measure = outputs * max_measure * ps
            predicted_measure = predicted_measure.squeeze(1)
            measure = measure.cuda(non_blocking=True)
            mae_value = mae(predicted_measure, measure)
            mape_value = mape(predicted_measure, measure)
            rmse_value = rmse(predicted_measure, measure)
            print(f'Predicted measure tensor: {predicted_measure}')
            print(f'Measure tensor: {measure}')
            #r2_value = r2_score(predicted_measure, measure)
            
            print(f"Predicted measure: {predicted_measure[0]}, real measure: {measure[0]}, mae: {mae_value}, mape: {mape_value}, rmse: {rmse_value}")
            
            mae_value = reduce_tensor(mae_value)
            mape_value = reduce_tensor(mape_value)
            rmse_value = reduce_tensor(rmse_value)
            #r2_value = reduce_tensor(r2_value)
            
            mae_meter.update(mae_value)
            mape_meter.update(mape_value)
            rmse_meter.update(rmse_value)
            #r2_meter.update(r2_value)
            
        if config.MODEL.TASK_TYPE == 'cls':
            acc1 = accuracy(outputs, Class, num_classes=config.MODEL.NUM_CLASSES, average = 'micro', 
                            mdmc_average = 'global' if config.PARALLEL_TYPE == 'model_parallel' else None)
            precision, recall = precision_recall(outputs, Class, num_classes=config.MODEL.NUM_CLASSES, average = 'micro',
                                                 mdmc_average = 'global' if config.PARALLEL_TYPE == 'model_parallel' else None)
            
            f1 = f1_score(outputs, Class, num_classes=config.MODEL.NUM_CLASSES, average = 'micro', 
                          mdmc_average = 'global' if config.PARALLEL_TYPE == 'model_parallel' else None)
            
            
            
            acc1 = reduce_tensor(acc1)
            f1 = reduce_tensor(f1)
            precision = reduce_tensor(precision)
            recall = reduce_tensor(recall)
            loss_cls = reduce_tensor(loss_cls)
            loss_meter_cls.update(loss_cls.item())
            acc1_meter.update(acc1.item())
            precision_meter.update(precision.item())
            recall_meter.update(recall.item())
            f1_score_meter.update(f1.item())
            
        elif config.MODEL.TASK_TYPE == 'reg':
            loss_meter_reg.update(loss.item())


        if idx % config.PRINT_FREQ == 0:
            if config.MODEL.TASK_TYPE == 'cls':
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'cls Loss {loss_meter_cls.val:.4f} ({loss_meter_cls.avg:.4f})\t' 
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t' 
                    f'f@1_score {f1_score_meter.val:.3f} ({f1_score_meter.avg:.3f})\t'
                    f'recall {recall_meter.val:.3f} ({recall_meter.avg:.3f})\t' 
                    f'precision {precision_meter.val:.3f} ({precision_meter.avg:.3f})\t')
            elif config.MODEL.TASK_TYPE == 'reg':
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'reg Loss {loss_meter_reg.val:.4f} ({loss_meter_reg.avg:.4f})\t' 
                    f'mae {mae_meter.val:.3f} ({mae_meter.avg:.3f})\t' 
                    f'mape {mape_meter.val:.3f} ({mape_meter.avg:.3f})\t' 
                    f'rmse {rmse_meter.val:.3f} ({rmse_meter.avg:.3f})\t')
    if config.MODEL.TASK_TYPE == 'cls':
        print('Finished validation! Results:')
        logger.info(f' * Acc@1 {acc1_meter.avg:.3f}')
        logger.info(f' * f@1_score {f1_score_meter.avg:.3f}')
        logger.info(f' * recall {recall_meter.avg:.3f}')
        logger.info(f' * precision {precision_meter.avg:.3f}')
        logger.info(f' * cls Loss {loss_meter_cls.avg:.4f}')
        experiment.log_metric('Val acc1', acc1_meter.avg, epoch=epoch)
        experiment.log_metric('Val f1_score', f1_score_meter.avg, epoch=epoch)
        experiment.log_metric('Val recall', recall_meter.avg, epoch=epoch)
        experiment.log_metric('Val precision', precision_meter.avg, epoch=epoch)
        experiment.log_metric('Val loss', loss_meter_cls.avg, epoch=epoch)
    elif config.MODEL.TASK_TYPE == 'reg':
        print('Finished validation! Results:')
        logger.info(f' * mae {mae_meter.avg:.3f}')
        logger.info(f' * mape {mape_meter.avg:.3f}')
        logger.info(f' * rmse {rmse_meter.avg:.3f}')
        logger.info(f' * Reg loss {loss_meter_reg.avg:.3f}')
        experiment.log_metric("Val mae", mae_meter.avg)
        experiment.log_metric("Val mape", mape_meter.avg)
        experiment.log_metric("Val rmse", rmse_meter.avg)
        experiment.log_metric("Validation reg loss", loss_meter_reg.avg)
            
    if config.MODEL.TASK_TYPE == 'cls':
        return acc1_meter.avg, f1_score_meter.avg, recall_meter.avg, precision_meter.avg, loss_meter_cls.avg
    elif config.MODEL.TASK_TYPE == 'reg':
        return mae_meter.avg, mape_meter.avg, rmse_meter.avg, loss_meter_reg.avg




if __name__ == '__main__':
    args, config = parse_option()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if config.PARALLEL_TYPE == 'ddp':
        torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 224
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 224
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 224

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))


    main(config)
