'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-01-04 23:51:49
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2023-02-23 10:33:33
'''
'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-01-04 23:51:49
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2023-02-18 17:44:08
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

import pdb
import os

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..da_heads.da_heads import build_da_heads, build_da_heads_triplet
from ..detector.attention import CAM_Module, PAM_Module


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.da_heads = build_da_heads(cfg)


        self.triplet_use=cfg.MODEL.DA_HEADS.TRIPLET_USE
        self.da_heads_triplet = build_da_heads_triplet(cfg) if self.triplet_use else False
        
        self.Aligned = cfg.MODEL.DA_HEADS.ALIGNMENT

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)

        features = self.backbone(images.tensors)

        proposals, proposal_losses = self.rpn(images, features, targets)

        da_losses = {}
        if self.roi_heads:

                if self.training: ### for Domain Adaptation loss 

                    fea_s = []
                    fea_p = []
                    fea_n = []
                    fea_s.append(features[0][0:1].clone())
                    fea_p.append(features[0][1:2].clone())
                    fea_n.append(features[0][2:3].clone())
                    da_img_fea_set = [fea_s, fea_p, fea_n]

                    ### the original components
                    ori_features = []
                    ori_features.append(features[0][0:2].clone())
                    ori_proposals = proposals[0:2]
                    ori_targets = targets[0:2]

                    x, result, detector_losses, da_ins_feas, da_ins_labels = self.roi_heads(ori_features, ori_proposals, ori_targets)

                    if self.da_heads_triplet:
                                                                                        
                        ###for triplet loss from three proposals
                        x_, result_, detector_losses_, da_ins_feas_s, da_ins_labels_ = self.roi_heads(fea_s, [proposals[0]], [targets[0]])####Source domain

                        if self.Aligned:
                            x_, result_, detector_losses_, da_ins_feas_p, da_ins_labels_ = self.roi_heads(fea_p, [proposals[0]], [targets[1]])####Target domain
                            x_, result_, detector_losses_, da_ins_feas_n, da_ins_labels_ = self.roi_heads(fea_n, [proposals[0]], [targets[2]])####Negative domain
                        else:
                            x_, result_, detector_losses_, da_ins_feas_p, da_ins_labels_ = self.roi_heads(fea_p, [proposals[1]], [targets[1]])####Target domain
                            x_, result_, detector_losses_, da_ins_feas_n, da_ins_labels_ = self.roi_heads(fea_n, [proposals[2]], [targets[2]])####Negative domain
                        
                        
                        da_ins_feas_set = [da_ins_feas_s, da_ins_feas_p, da_ins_feas_n]
                        da_losses = self.da_heads_triplet(ori_features, da_ins_feas, da_ins_labels, da_ins_feas_set, da_img_fea_set, ori_targets)### add the da_ins_feas_set

                    elif self.da_heads:
                        da_losses = self.da_heads(features, da_ins_feas, da_ins_labels, targets)



                else:# Original Loss

                    x, result, detector_losses, da_ins_feas, da_ins_labels = self.roi_heads(features, proposals, targets)
        



        else:
            # RPN-only models don't have roi_heads

            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(da_losses)
            return losses
    
        torch.cuda.empty_cache()
        return result
