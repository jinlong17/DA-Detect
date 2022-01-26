# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer

#TODO: jinlong
from .loss import make_da_heads_loss_evaluator, triplet_loss_module

import pdb

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(DomainAdaptationModule, self).__init__()

        self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith('V') else res2_out_channels * stage2_relative_factor
        
        self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        
        self.img_weight = cfg.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT
        self.ins_weight = cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT
        self.cst_weight = cfg.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)
        #TODO:jinlong
        self.loss_triplet = triplet_loss_module

    def forward(self, img_features, da_ins_feature, da_ins_labels, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        if self.resnet_backbone:
            # pdb.set_trace()
            da_ins_feature = self.avgpool(da_ins_feature)

        # pdb.set_trace()
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)

        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
        # pdb.set_trace()
        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)

        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()


        # pdb.set_trace()

        
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
            )




            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss

            if self.ins_weight > 0:
                losses["loss_da_instance"] = self.ins_weight * da_ins_loss

            if self.cst_weight > 0:
                losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            return losses
        return {}

def build_da_heads(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule(cfg)
    return []




class DomainAdaptationModule_triplet(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(DomainAdaptationModule_triplet, self).__init__()

        self.cfg = cfg.clone()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith('V') else res2_out_channels * stage2_relative_factor
        
        self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        
        self.img_weight = cfg.MODEL.DA_HEADS.DA_IMG_LOSS_WEIGHT
        self.ins_weight = cfg.MODEL.DA_HEADS.DA_INS_LOSS_WEIGHT
        self.cst_weight = cfg.MODEL.DA_HEADS.DA_CST_LOSS_WEIGHT

        self.grl_img = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)
        #TODO:jinlong
        self.loss_triplet = triplet_loss_module

    def forward(self, img_features, da_ins_feature, da_ins_labels, da_ins_feas_set, img_fea_set, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)
            TODO: jinlong add triplet loss here
            TODO: jinlong img_fea_set(list[Tensor): sourse domain, positive target, negative target
            TODO: jinlong da_ins_feas_set(list[Tensor): sourse domain, positive target, negative target

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        da_ins_fea_s = da_ins_feas_set[0]
        da_ins_fea_p = da_ins_feas_set[1]
        da_ins_fea_n = da_ins_feas_set[2]
        img_features_s = img_fea_set[0]
        img_features_p = img_fea_set[1]
        img_features_n = img_fea_set[2]


        if self.resnet_backbone:
            # pdb.set_trace()
            da_ins_feature = self.avgpool(da_ins_feature)
            #TODO:jinlong
            da_ins_fea_s = self.avgpool(da_ins_fea_s)
            da_ins_fea_p = self.avgpool(da_ins_fea_p)
            da_ins_fea_n = self.avgpool(da_ins_fea_n)

        # pdb.set_trace()
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)
        #TODO:jinlong
        da_ins_fea_s = da_ins_fea_s.view(da_ins_fea_s.size(0), -1)
        da_ins_fea_p = da_ins_fea_p.view(da_ins_fea_p.size(0), -1)
        da_ins_fea_n = da_ins_fea_n.view(da_ins_fea_n.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        #TODO:jinlong
        # img_grl_fea_s = [self.grl_img(fea) for fea in img_features_s]
        # img_grl_fea_p = [self.grl_img(fea) for fea in img_features_p]
        # img_grl_fea_n = [self.grl_img(fea) for fea in img_features_n]
        # ins_grl_fea_s = self.grl_ins(da_ins_fea_s)
        # ins_grl_fea_p = self.grl_ins(da_ins_fea_p)
        # ins_grl_fea_n = self.grl_ins(da_ins_fea_n)

        img_grl_fea_s = [self.grl_img_consist(fea) for fea in img_features_s]
        img_grl_fea_p = [self.grl_img_consist(fea) for fea in img_features_p]
        img_grl_fea_n = [self.grl_img_consist(fea) for fea in img_features_n]
        ins_grl_fea_s = self.grl_ins_consist(da_ins_fea_s)
        ins_grl_fea_p = self.grl_ins_consist(da_ins_fea_p)
        ins_grl_fea_n = self.grl_ins_consist(da_ins_fea_n)


        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
        # pdb.set_trace()
        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        #TODO:jinlong
        da_img_features_s = self.imghead(img_grl_fea_s)
        da_img_features_p = self.imghead(img_grl_fea_p)
        da_img_features_n = self.imghead(img_grl_fea_n)
        da_ins_features_s = self.inshead(ins_grl_fea_s)
        da_ins_features_p = self.inshead(ins_grl_fea_p)
        da_ins_features_n = self.inshead(ins_grl_fea_n)
        # pdb.set_trace()
        # da_img_features_s = img_grl_fea_s
        # da_img_features_p = img_grl_fea_p
        # da_img_features_n = img_grl_fea_n
        # da_ins_features_s = ins_grl_fea_s
        # da_ins_features_p = ins_grl_fea_p
        # da_ins_features_n = ins_grl_fea_n




        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()


        # pdb.set_trace()

        
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
                da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
            )
            #TODO: jinlong
            da_triplet_img_loss = self.loss_triplet(da_img_features_s[0], da_img_features_p[0], da_img_features_n[0])
            da_triplet_ins_loss = self.loss_triplet(da_ins_features_s, da_ins_features_p, da_ins_features_n)



            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
                # losses["triplet_loss_da_image"] = self.img_weight * da_triplet_img_loss#TODO: jinlong
            if self.ins_weight > 0:
                losses["loss_da_instance"] = self.ins_weight * da_ins_loss
                # losses["triplet_loss_da_instance"] = self.ins_weight * da_triplet_ins_loss#TODO: jinlong
            # if self.img_weight > 0:
            #     losses["loss_da_image"] = self.img_weight* da_img_loss/da_img_loss.detach()*0.1
            #     losses["triplet_loss_da_image"] =  self.img_weight*da_triplet_img_loss/da_triplet_img_loss.detach()*0.1#TODO: jinlong
            # if self.ins_weight > 0:
            #     losses["loss_da_instance"] = self.ins_weight* da_ins_loss/da_ins_loss.detach()*0.1
            #     losses["triplet_loss_da_instance"] = self.ins_weight* da_triplet_ins_loss/da_triplet_ins_loss.detach()*0.1 #TODO: jinlong



            # if self.cst_weight > 0:
            #     losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss
            # return losses
        return {}

def build_da_heads_triplet(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule_triplet(cfg)
    return []