
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator, BatchCollator_triplet
from .transforms import build_transforms

import pdb
import numpy as np
import random
import copy


class Dataset_triplet(torch.utils.data.Dataset):

        def __init__(self, datasets):
            super(Dataset_triplet, self).__init__()

            self.dataset_s = datasets[0]
            self.dataset_p = datasets[1]
            self.dataset_n = datasets[2]

        def __getitem__(self, index):

            img_s, target_s, idx1 = self.dataset_s.__getitem__(index)
            img_p, target_p, idx2 = self.dataset_p.__getitem__(index)
            img_n, target_n, idx3 = self.dataset_n.__getitem__(index)



            target_p_same = copy.deepcopy(target_s)
            domain_labels = copy.deepcopy(target_p.get_field("is_source"))
            target_p_same.add_field("is_source", domain_labels)
            
            target_n_same = copy.deepcopy(target_s)
            domain_labels1 = copy.deepcopy(target_n.get_field("is_source"))
            target_n_same.add_field("is_source", domain_labels1)



            # print('idx1: ', idx1,"idx2: ", idx2, "idx3: ", idx3)
            return img_s, target_s, img_p, target_p_same, img_n, target_n_same, idx1, idx2, idx3
            # return img_s, target_s, img_p, target_p_same, img_n, target_n_same
        

        def __len__(self):
            return len(self.dataset_s)

        def get_img_info(self, index):
            # img_id = self.id_to_img_map[index]
            # img_data = self.coco.imgs[img_id]
            # return img_data
            return self.dataset_s.get_img_info(index)




def build_dataset_da(dataset_list, transforms, dataset_catalog, is_source, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []

    for i in range(len(dataset_list)):
        dataset_name = dataset_list[i]
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]

        # for COCODataset, we want to remove images without annotations
        # during training

        #TODO: jinlong
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # pdb.set_trace()
        args["is_source"] = is_source[i]
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)


    #TODO: jinlong
    # pdb.set_trace()
    final_dataset = Dataset_triplet(datasets)

    # for testing, return a list of datasets

    #TODO: jinlong
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    # dataset = datasets[0]
    # if len(datasets) > 1:
    #     dataset = D.ConcatDataset(datasets)

    # return [dataset]
    return [final_dataset]


def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True, is_source=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    # root_set = []
    # ann_file_set = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]

        # args["root_set"] = copy.deepcopy(args["root"])
        # args["ann_file_set"] = copy.deepcopy(args["ann_file"])
        # root_set.append(copy.deepcopy(args["root_set"]))
        # ann_file_set.append(copy.deepcopy(args["ann_file_set"]))

        # del args["root"]
        # del args["ann_file"]

    # args["root_set"] = copy.deepcopy(root_set)
    # args["ann_file_set"] = copy.deepcopy(ann_file_set)

        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        # if data["factory"] == "COCODataset_da":
        #     args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        args["is_source"] = is_source
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)


    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_source=True, is_negative=False, is_distributed=False, is_for_period=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        if cfg.MODEL.DOMAIN_ADAPTATION_ON:
            assert (
            images_per_batch % (2*num_gpus) == 0
            ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by 2 times the number "
            "of GPUs ({}) used.".format(images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // (2*num_gpus)



        shuffle = True ##TODO: jinlong

        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True 
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if is_train:
        if cfg.MODEL.DOMAIN_ADAPTATION_ON:

            if is_source:
    
                dataset_list = cfg.DATASETS.SOURCE_TRAIN 
            else:
                if is_negative:
                    dataset_list= cfg.DATASETS.TARGET_TRAIN_negative
                else:
                    dataset_list= cfg.DATASETS.TARGET_TRAIN

        else:
            dataset_list = cfg.DATASETS.TRAIN
            
            # dataset_list = [cfg.DATASETS.SOURCE_TRAIN[0],cfg.DATASETS.TARGET_TRAIN[0], cfg.DATASETS.TARGET_TRAIN_negative[0]]
            # dataset_list = cfg.DATASETS.SOURCE_TRAIN
    else:
        dataset_list = cfg.DATASETS.TEST

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog,  is_train, is_source)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


#TODO: jinlong
def make_data_loader_da(cfg, is_source, is_train=True, is_negative=False, is_distributed=False, is_for_period=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        if cfg.MODEL.DOMAIN_ADAPTATION_ON:
            assert (
            images_per_batch % (2*num_gpus) == 0
            ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by 2 times the number "
            "of GPUs ({}) used.".format(images_per_batch, num_gpus)
            images_per_gpu = images_per_batch // (2*num_gpus)



        shuffle = True ##TODO: jinlong

        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True 
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if is_train:
        if cfg.MODEL.DOMAIN_ADAPTATION_ON:
            #TODO:jinlong
            dataset_list = [cfg.DATASETS.SOURCE_TRAIN[0],cfg.DATASETS.TARGET_TRAIN[0], cfg.DATASETS.TARGET_TRAIN_negative[0]]
    else:
        #TODO:jinlong
        dataset_list = cfg.DATASETS.TEST
        # dataset_list = [cfg.DATASETS.TEST[0], cfg.DATASETS.TEST_SOURCE[0], cfg.DATASETS.TEST_SOURCE[0]]

    transforms = build_transforms(cfg, is_train)
    # pdb.set_trace()
    datasets = build_dataset_da(dataset_list, transforms, DatasetCatalog,is_source, is_train or is_for_period)

    data_loaders = []
    for dataset in datasets:
        # pdb.set_trace()
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BatchCollator_triplet(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

        data_loaders.append(data_loader)
    if is_train or is_for_period:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders