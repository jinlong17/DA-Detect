# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer

#TODO: jinlong
from .loss import make_da_heads_loss_evaluator, make_da_heads_loss_evaluator_original
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
        self.in_channels=in_channels

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x



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
        self.triplet_img_weight = cfg.MODEL.DA_HEADS.DA_TRIPLET_IMG_WEIGHT
        self.triplet_ins_weight = cfg.MODEL.DA_HEADS.DA_TRIPLET_INS_WEIGHT


        self.grl_img = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        self.grl_img_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        self.grl_ins_consist = GradientScalarLayer(1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT)
        
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)

        # self.loss_triplet = triplet_loss_module
        self.ins_loss = [1]
        self.img_loss = [1]
        self.triplet_ins = [1]
        self.triplet_img = [1]


        self.advGRL=cfg.MODEL.DA_HEADS.DA_ADV_GRL
        self.advGRL_threshold = cfg.MODEL.DA_HEADS.DA_ADV_GRL_THRESHOLD
        self.triplet_metric_img = cfg.MODEL.DA_HEADS.TRIPLET_MARGIN_IMG
        self.triplet_metric_ins = cfg.MODEL.DA_HEADS.TRIPLET_MARGIN_INS
        self.triplet_max_margin = cfg.MODEL.DA_HEADS.TRIPLET_MAX_MARGIN

        self.triplet_loss = self.loss_evaluator.triplet_loss
        

    def DA_Img_component(self,img_features,targets):

        #clalcute the current loss for advGRL first
        current_feature = self.imghead(img_features)
        current_feature = [fea.detach() for fea in current_feature]
        current_loss = self.loss_evaluator.da_img_loss(current_feature, targets)

        if self.advGRL:
            # img_grl_fea = self.Adv_GRL_Optimized(current_loss,img_features)
            img_grl_fea = self.Adv_GRL(current_loss,img_features)
        else:
            img_grl_fea = [self.grl_img(fea) for fea in img_features]

        da_img_features = self.imghead(img_grl_fea)

        # print("da_img_features is used ")
        da_img_loss = self.loss_evaluator.da_img_loss(da_img_features, targets)

        return da_img_loss



    def DA_Ins_component(self,da_ins_feature,da_ins_labels):

        if self.resnet_backbone:
            da_ins_feature = self.avgpool(da_ins_feature)
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        #clalcute the current loss for advGRL first
        current_features = self.inshead(da_ins_feature.detach())
        current_loss = self.loss_evaluator.da_ins_loss(current_features, da_ins_labels)
        
        if self.advGRL:
            # ins_grl_fea = self.Adv_GRL_Optimized(current_loss, da_ins_feature,list_option=False)
            ins_grl_fea = self.Adv_GRL(current_loss, da_ins_feature,list_option=False)
        else:
            ins_grl_fea = self.grl_ins(da_ins_feature)

        da_ins_features = self.inshead(ins_grl_fea)
        da_ins_loss = self.loss_evaluator.da_ins_loss(da_ins_features, da_ins_labels)


        # print("da_ins_features is used")

        return da_ins_loss 


    
    def Adv_GRL(self,loss_iter, input_features, list_option=True):

        self.bce = F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.7,0.3]]), torch.FloatTensor([[1,0]]))

        if loss_iter <=  self.bce:
            adv_threshold = min(self.advGRL_threshold, 1/loss_iter)
            # self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold)

            if list_option:# for img_features (list[Tensor])
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold.numpy())
                advgrl_fea = [ self.advGRL_optimized(fea) for fea in input_features]
            else: # for da_ins_feature (Tensor)
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_advGRL_WEIGHT*adv_threshold.numpy())
                advgrl_fea = self.advGRL_optimized(input_features)
        else:
            if list_option:# for img_features (list[Tensor])
                advgrl_fea = [self.grl_img(fea) for fea in input_features]###the original component
            else: # for da_ins_feature (Tensor)
                advgrl_fea = self.grl_ins(input_features)

        # print("Adv_GRL is used")
        
        return advgrl_fea
    
    def Adv_GRL_Optimized(self,loss_iter, input_features, list_option=True):

        self.bce_min = F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.6,0.4]]), torch.FloatTensor([[1,0]]))#0.6288
        self.bce_max = F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.55,0.45]]), torch.FloatTensor([[1,0]]))#0.6753
        #[0.5, 0.5] = 0.7241, [0.9, 0.1] = 0.5428

        if loss_iter <=  self.bce_min:

            adv_threshold = min(self.advGRL_threshold, 1/loss_iter)
            # self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold)

            if list_option:# for img_features (list[Tensor])
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold)
                advgrl_fea = [ self.advGRL_optimized(fea) for fea in input_features]
            else: # for da_ins_feature (Tensor)
                self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_advGRL_WEIGHT*adv_threshold)
                advgrl_fea = self.advGRL_optimized(input_features)

        elif loss_iter >=  self.bce_max:

            adv_threshold = torch.tensor(0.1)
            self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold)

            if list_option:# for img_features (list[Tensor])
                # self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_advGRL_WEIGHT*adv_threshold.numpy())
                advgrl_fea = [ self.advGRL_optimized(fea) for fea in input_features]
            else: # for da_ins_feature (Tensor)
                # self.advGRL_optimized = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_advGRL_WEIGHT*adv_threshold.numpy())
                advgrl_fea = self.advGRL_optimized(input_features)
        else:
            if list_option:# for img_features (list[Tensor])
                advgrl_fea = [self.grl_img(fea) for fea in input_features]###the original component
            else: # for da_ins_feature (Tensor)
                advgrl_fea = self.grl_ins(input_features)

        # print("Adv_GRL is used")
        
        return advgrl_fea

    def Domainlevel_Img_component(self,img_fea_set):
        img_features_s = img_fea_set[0]
        img_features_p = img_fea_set[1]
        img_features_n = img_fea_set[2]
        #adaptive triplet loss
        da_triplet_img_loss = self.loss_evaluator.triplet_img_loss(img_features_s[0], img_features_p[0], img_features_n[0],self.triplet_img[-1], adaptive=True,lr=0.001, max_margin=self.triplet_max_margin, margin=self.triplet_metric_img)

        # da_triplet_img_loss = self.triplet_loss(img_features_s[0], img_features_p[0], img_features_n[0], margin=self.triplet_metric_img)

        # print("da_triplet_img_loss is used. ")
        # L1_loss = nn.L1Loss()
        # print("img_pos_dist is ",str(L1_loss(img_features_s[0],img_features_p[0]).detach().cpu().item())," img_neg_dist is ",str(L1_loss(img_features_s[0],img_features_n[0]).detach().cpu().item()))

        return da_triplet_img_loss

    def Domainlevel_Ins_component(self,da_ins_fea_set):

        da_ins_fea_s = da_ins_fea_set[0]
        da_ins_fea_p = da_ins_fea_set[1]
        da_ins_fea_n = da_ins_fea_set[2]
        if self.resnet_backbone:
            da_ins_fea_s = self.avgpool(da_ins_fea_s)
            da_ins_fea_p = self.avgpool(da_ins_fea_p)
            da_ins_fea_n = self.avgpool(da_ins_fea_n)

        da_ins_fea_s = da_ins_fea_s.view(da_ins_fea_s.size(0), -1)
        da_ins_fea_p = da_ins_fea_p.view(da_ins_fea_p.size(0), -1)
        da_ins_fea_n = da_ins_fea_n.view(da_ins_fea_n.size(0), -1)

        #adaptive triplet loss
        da_triplet_ins_loss = self.loss_evaluator.triplet_ins_loss(da_ins_fea_s, da_ins_fea_p, da_ins_fea_n,self.triplet_ins[-1], adaptive=False,lr=0.001, max_margin=self.triplet_max_margin, margin=self.triplet_metric_ins)

        # da_triplet_ins_loss = self.triplet_loss(da_ins_fea_s, da_ins_fea_p, da_ins_fea_n, margin=self.triplet_metric_ins)

        # print("Ins_pos_dist is ",str((da_ins_fea_s-da_ins_fea_p).pow(2).sum(1).item()), "Ins_neg_dist is ",str((da_ins_fea_s-da_ins_fea_n).pow(2).sum(1).item()))

        # print("da_triplet_ins_loss is used. ")

        return da_triplet_ins_loss
    
    def Consistency_component(self,img_features, da_ins_feature,da_ins_labels):

        if self.resnet_backbone:
            da_ins_feature = self.avgpool(da_ins_feature)

        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]###the original component
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)###the original component
        da_img_consist_features = self.imghead(img_grl_consist_fea)###the original component
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)###the original component
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]###the original component
        da_ins_consist_features = da_ins_consist_features.sigmoid()###the original component

        da_consistency_loss = self.loss_evaluator.da_consist_loss(da_img_consist_features, da_ins_consist_features,da_ins_labels)
        return da_consistency_loss

    def forward(self, img_features, da_ins_feature, da_ins_labels, da_ins_feas_set, img_fea_set, targets=None):
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

        if self.training:

            ###the original component
            # da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator(
            #     da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, targets
            # )

            ###the triplet loss for img and ins

            losses = {}

            if self.triplet_ins_weight> 0:######triplet loss of ins
                da_triplet_ins_loss = self.Domainlevel_Ins_component(da_ins_feas_set)
                losses["triplet_loss_instance"] = self.triplet_ins_weight * da_triplet_ins_loss
                self.triplet_ins.append(da_triplet_ins_loss.cpu().detach())# log the for triplet loss of ins

            if self.triplet_img_weight > 0:######triplet loss of img
                da_triplet_img_loss = self.Domainlevel_Img_component(img_fea_set)
                losses["triplet_loss_image"] = self.triplet_img_weight * da_triplet_img_loss
                self.triplet_img.append(da_triplet_img_loss.cpu().detach())# log the for triplet loss of img

            if self.img_weight > 0:
                ###the DA img loss
                da_img_loss = self.DA_Img_component(img_features,targets)
                losses["loss_da_image"] = self.img_weight * da_img_loss###the original component

            if self.ins_weight > 0:
                ###the DA ins loss
                da_ins_loss = self.DA_Ins_component(da_ins_feature,da_ins_labels)
                losses["loss_da_instance"] = self.ins_weight * da_ins_loss###the original component
                
            if self.cst_weight > 0:###the original component
                ###the consistency loss
                da_consistency_loss = self.Consistency_component(img_features, da_ins_feature,da_ins_labels)
                losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss


            return losses
        return {}

def build_da_heads_triplet(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule_triplet(cfg)
    return []



#### the Original DA module
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
        self.loss_evaluator_origin = make_da_heads_loss_evaluator_original(cfg)

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

            da_ins_feature = self.avgpool(da_ins_feature)


        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)

        img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]
        ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)

        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)

        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()

        
        if self.training:
            da_img_loss, da_ins_loss, da_consistency_loss = self.loss_evaluator_origin(
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