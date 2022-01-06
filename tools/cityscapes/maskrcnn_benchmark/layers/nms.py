'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-01-04 10:58:11
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-01-04 17:06:05
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
# from maskrcnn_benchmark import _C
from ._utils import _C

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
