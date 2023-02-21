'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-01-04 23:51:49
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-08-26 12:05:25
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

import pdb
import copy

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor, make_roi_box_post_processor_jinlong
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        #TODO:jinlong for Branch
        # self.post_processor_jinlong = make_roi_box_post_processor_jinlong(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        proposals_b = copy.deepcopy(proposals)
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        
        # final classifier that converts the features into predictions
        
        class_logits, box_regression = self.predictor(x)

        ###
        #TODO:jinlong for Branch
        # if self.training:
        #     if len(proposals) == 1:
        #         mask_branch_proposals = []
        #     else:
        #         mask_branch_proposals = []
        #         # fea_s = []
        #         # fea_p = []
        #         # fea_s.append(features[0][0:1].clone())
        #         # fea_p.append(features[0][1:2].clone())
        #         x_s = self.feature_extractor([features[0][0:1]], [proposals_b[0]])
        #         x_p = self.feature_extractor([features[0][1:2]], [proposals_b[1]])
        #         class_logits_s, box_regression_s = self.predictor(x_s)
        #         class_logits_p, box_regression_p = self.predictor(x_p)
        #         mask_branch_s = self.post_processor_jinlong((class_logits_s, box_regression_s), [proposals_b[0]])
        #         mask_branch_p = self.post_processor_jinlong((class_logits_p, box_regression_p), [proposals_b[1]])
        #         # mask_branch_s = self.post_processor((class_logits_s, box_regression_s), [proposals_b[0]])
        #         # mask_branch_p = self.post_processor((class_logits_p, box_regression_p), [proposals_b[1]])

        #         mask_branch_proposals = [mask_branch_s, mask_branch_p]
        # else:
        #     mask_branch_proposals = []
        ###jinlong for Branch


        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            
            return x, result, {}, x, None
            #TODO:jinlong
            # return x, result, {}, x, None, None

        loss_classifier, loss_box_reg, _ = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        if self.training:
            with torch.no_grad():
                da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)

        da_ins_feas = self.feature_extractor(features, da_proposals)
        class_logits, box_regression = self.predictor(da_ins_feas)
        _, _, da_ins_labels = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
            da_ins_feas,
            da_ins_labels
        )
        #TODO:jinlong: for the Branch
        # return (
        #     x,
        #     proposals,
        #     dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        #     da_ins_feas,
        #     da_ins_labels,
        #     mask_branch_proposals
        # )
  

def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
