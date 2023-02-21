'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-02-14 23:48:48
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-02-14 23:48:48
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import GradientScalarLayer

#TODO: jinlong
from .loss import make_da_heads_loss_evaluator, triplet_loss_module, PAM_Module, CAM_Module

import copy
import pdb
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from torchvision.utils import save_image

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
        #TODO:jinlong 
        self.ins_loss = [1]
        self.img_loss = [1]
        self.mask_loss= [1]


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
        self.PAM_img = PAM_Module(1024)
        self.PAM_ins = PAM_Module(2048)

    def forward(self, img_features, da_ins_feature, da_ins_labels, da_ins_feas_set, img_fea_set, targets=None):#mask_branch_proposals
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
        da_ins_fea_s_1 = da_ins_feas_set[0]
        da_ins_fea_p_1 = da_ins_feas_set[1]
        da_ins_fea_n_1 = da_ins_feas_set[2]
        img_features_s = img_fea_set[0]
        img_features_p = img_fea_set[1]
        img_features_n = img_fea_set[2]

        # w=38/600
        # h=76/1200
        # # #TODO:jinlong
        # img_branch_fea = [img_features[0].clone()]
        # for i in range(len(mask_branch_proposals)):

        #     result_nms = boxlist_nms(mask_branch_proposals[i][0], 0.3, max_proposals=20)
        #     img_block = torch.zeros_like(img_features[0][i])
            
        #     for ins in result_nms.bbox.cpu():

        #         ins = ins.detach().numpy()

        #         for a in range(len(img_block[1,:,1])):

        #             if  ins[1]*w <= a <=  ins[3]*w:

        #                 for  b in range(len(img_block[1,1,:])):

        #                     if  ins[0]*h <= b <= ins[2]*h:

        #                             img_block[:,a,b] = 1
        #     img_branch_fea[0][i] = copy.deepcopy(img_block.mul(img_features[0][i]))

            # save_image(img_block[1:4,:,:],'img'+str(i)+'.png')

        # pdb.set_trace()




        if self.resnet_backbone:
            # pdb.set_trace()
            da_ins_feature = self.avgpool(da_ins_feature)
            #TODO:jinlong
            da_ins_fea_s = self.avgpool(da_ins_fea_s_1)
            da_ins_fea_p = self.avgpool(da_ins_fea_p_1)
            da_ins_fea_n = self.avgpool(da_ins_fea_n_1)




        # pdb.set_trace()
        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)
        #TODO:jinlong
        da_ins_fea_s = da_ins_fea_s.view(da_ins_fea_s.size(0), -1)
        da_ins_fea_p = da_ins_fea_p.view(da_ins_fea_p.size(0), -1)
        da_ins_fea_n = da_ins_fea_n.view(da_ins_fea_n.size(0), -1)

        # if self.img_loss[-1] <= F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.7]]), torch.FloatTensor([[0.3]])):
        #     threshold_img = min(30, 1/self.img_loss[-1])
        #     if threshold_ins== 30:
        #         print("threshold_ins: ", threshold_ins)
        #     self.grl_img_optimize = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT*threshold_img.numpy())
        #     img_grl_fea = [self.grl_img_optimize(fea) for fea in img_features]
        #     # print('img_grl: ', threshold_img.numpy())
            
        # else:
        #     img_grl_fea = [self.grl_img(fea) for fea in img_features]
            

        # if self.ins_loss[-1] <= F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.7]]), torch.FloatTensor([[0.3]])):
        #     threshold_ins = min(30, 1/self.ins_loss[-1])
        #     if threshold_ins== 30:
        #         print("threshold_ins: ", threshold_ins)
        #     self.grl_ins_optimize = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT*threshold_ins.numpy())
        #     ins_grl_fea = self.grl_ins_optimize(da_ins_feature)
        #     # print('ins_grl: ', threshold_ins.numpy())
        # else:
        #     ins_grl_fea = self.grl_ins(da_ins_feature)


        # if self.mask_loss[-1] <= F.binary_cross_entropy_with_logits(torch.FloatTensor([[0.7]]), torch.FloatTensor([[0.3]])):
        #     threshold_img = min(30, 1/self.mask_loss[-1])
        #     self.grl_img_optimize = GradientScalarLayer(-1.0*self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT*threshold_img.numpy())
        #     img_grl_fea = [self.grl_img_optimize(fea) for fea in img_branch_fea]
            
        # else:
        #     img_grl_fea = [self.grl_img(fea) for fea in img_branch_fea]

        # da_mask_features = self.imghead(img_branch_fea)



        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)
        #TODO:jinlong

        # img_grl_fea_s = [self.grl_img_consist(fea) for fea in img_features_s]
        # img_grl_fea_p = [self.grl_img_consist(fea) for fea in img_features_p]
        # img_grl_fea_n = [self.grl_img_consist(fea) for fea in img_features_n]
        # ins_grl_fea_s = self.grl_ins_consist(da_ins_fea_s)
        # ins_grl_fea_p = self.grl_ins_consist(da_ins_fea_p)
        # ins_grl_fea_n = self.grl_ins_consist(da_ins_fea_n)


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
            # da_img_loss, da_ins_loss, da_consistency_loss, da_mask_loss = self.loss_evaluator(
            #     da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, da_ins_labels, da_mask_features,targets
            # )
            #TODO: jinlong
            m=1.0
            da_triplet_img_loss = self.loss_triplet(img_features_s[0], img_features_p[0], img_features_n[0],m)
            da_triplet_ins_loss = self.loss_triplet(da_ins_fea_s, da_ins_fea_p, da_ins_fea_n, m)


            # da_triplet_ins_loss = self.loss_triplet(da_ins_fea_s_1, da_ins_fea_p_1, da_ins_fea_n_1, m)

            # attention loss
            # pdb.set_trace()
            # img_attention_s = self.PAM_img(img_features_s[0])
            # img_attention_p = self.PAM_img(img_features_p[0])

            # ins_attention_s = self.PAM_ins(da_ins_fea_s_1)
            # ins_attention_p = self.PAM_ins(da_ins_fea_p_1)

            # loss_MSE = torch.nn.MSELoss(reduce=True, size_average=True)

            # pam_img_attention_loss = loss_MSE(img_attention_s,img_attention_p)
            # pam_ins_attention_loss = loss_MSE(ins_attention_s,ins_attention_p)

            # cam_img_attention_loss = loss_MSE(img_attention_s_c,img_attention_p_c)
            

            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
                losses["triplet_loss_da_image"] = self.img_weight * da_triplet_img_loss
                # losses["da_mask_loss"] = self.img_weight * da_mask_loss

                # losses["img_pam_loss"] = self.img_weight * pam_img_attention_loss
                # losses["ins_pam_loss"] = self.img_weight * pam_ins_attention_loss
                # losses["img_cam_loss"] = self.img_weight * cam_img_attention_loss

                self.img_loss.append(da_img_loss.cpu().detach())
                # self.img_loss.append(self.img_weight *da_img_loss.cpu().detach().numpy())


            if self.ins_weight > 0:
                losses["loss_da_instance"] = self.ins_weight * da_ins_loss
                losses["triplet_loss_da_instance"] = self.ins_weight * da_triplet_ins_loss

                self.ins_loss.append(da_ins_loss.cpu().detach())
                # self.ins_loss.append(self.ins_weight*da_ins_loss.cpu().detach().numpy())


            # if self.cst_weight > 0:
            #     losses["loss_da_consistency"] = self.cst_weight * da_consistency_loss



            return losses
        return {}

def build_da_heads_triplet(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule_triplet(cfg)
    return []