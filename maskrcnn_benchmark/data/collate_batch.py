'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-01-04 23:51:49
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-01-20 18:12:52
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list

import pdb

class BatchCollator_triplet(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        # img_ids = transposed_batch[2]
        images_p = to_image_list(transposed_batch[2], self.size_divisible)
        targets_p = transposed_batch[3]
        images_n = to_image_list(transposed_batch[4], self.size_divisible)
        targets_n = transposed_batch[5]
        # pdb.set_trace()
        idx1 = transposed_batch[6]
        idx2 = transposed_batch[7]
        idx3 = transposed_batch[8]
        return images, targets, images_p, targets_p, images_n, targets_n, idx1, idx2, idx3


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # pdb.set_trace()
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids