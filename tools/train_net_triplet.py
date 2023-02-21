'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-01-18 18:28:34
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2023-02-20 17:13:00
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader, make_data_loader_da
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train, do_da_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.solver import make_lr_scheduler 
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.metric_logger import (
    MetricLogger, TensorboardLogger)



# import warnings
import numpy as np
import random

import pdb

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



# def train(cfg, local_rank, distributed):
def train(cfg, local_rank, distributed, use_tensorboard=False):

    setup_seed(100)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.MODEL.OUTPUT_DIR

    save_to_disk = get_rank() == 0

    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=cfg.TENSORBOARD_EXPERIMENT,
            start_iter=arguments['iteration'],
            delimiter="  ")
    else:
        meters = MetricLogger(delimiter="  ")

    if checkpoint_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=False)
    else:
        data_loader_val = None

    if cfg.MODEL.DOMAIN_ADAPTATION_ON:

        triplet_data_loading = cfg.MODEL.DA_HEADS.TRIPLET_USE
        triplet_data_aligned = cfg.MODEL.DA_HEADS.ALIGNMENT

        if triplet_data_loading:
            if triplet_data_aligned:
                Positive_target_data_loader = make_data_loader_da(
                    cfg,
                    is_train=True,
                    is_source=[True, False, False],## source, target, negative
                    is_negative=False,
                    is_distributed=distributed,
                    start_iter=arguments["iteration"],
                )
                source_data_loader=[]
                Negative_target_data_loader=[]
            else:
                source_data_loader = make_data_loader(
                    cfg,
                    is_train=True,
                    is_source=True,
                    is_negative=False,
                    is_distributed=distributed,
                    start_iter=arguments["iteration"],
                )
                Negative_target_data_loader = make_data_loader(
                    cfg,
                    is_train=True,
                    is_source=False,
                    is_negative=True,
                    is_distributed=distributed,
                    start_iter=arguments["iteration"],
                )
                Positive_target_data_loader = make_data_loader(
                    cfg,
                    is_train=True,
                    is_source=False,
                    is_negative=False,
                    is_distributed=distributed,
                    start_iter=arguments["iteration"],
                )
            do_da_train(
                model,
                source_data_loader,
                Positive_target_data_loader,
                Negative_target_data_loader,
                data_loader_val,
                optimizer,
                scheduler,
                checkpointer,
                device,
                checkpoint_period,
                arguments,
                cfg,
                distributed,
                meters,
                triplet_data_loading=triplet_data_loading,
                triplet_data_aligned=triplet_data_aligned
            )
    else:
        data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        
        do_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )

    return model





def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  #check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()



def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    # parser.add_argument(
    #     "--output_dir",
    #     default="model",#img+ins+triple
    #     help="create the file name to save the model file",
    #     type=str,
    # )

    parser.add_argument(
        "--use-tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX installed)",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)

    cfg.MODEL.OUTPUT_DIR = os.path.join(cfg.MODEL.OUTPUT_DIR,cfg.MODEL.OUTPUT_SAVE_NAME)

    cfg.merge_from_list(args.opts)

    if cfg.MODEL.OUTPUT_DIR:
        mkdir(cfg.MODEL.OUTPUT_DIR)

    logger = setup_logger("maskrcnn_benchmark", cfg.MODEL.OUTPUT_DIR, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # model = train(cfg, args.local_rank, args.distributed)
    model = train(
        cfg=cfg,
        local_rank=args.local_rank,
        distributed=args.distributed,
        use_tensorboard=args.use_tensorboard
    )

    if not args.skip_test:
        cfg.OUTPUT_DIR = './'
        test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
