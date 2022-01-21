# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

from pycocotools.coco import COCO
from PIL import Image
import os

import pdb

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, is_source= True
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # pdb.set_trace()

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms
        self.is_source = is_source

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        # print('root: ', self.root)

        # filter crowd annotations
        # TODO: might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # print('boxes: ', boxes)
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        domain_labels = torch.ones_like(classes, dtype=torch.bool) if self.is_source else torch.zeros_like(classes, dtype=torch.bool)
        target.add_field("is_source", domain_labels)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # pdb.set_trace()
        # print('img_ids: ',self.ids[idx], ' idx: ', idx, ' anno: ',len(anno))
        # print('path: ', os.path.join(self.root, self.coco.loadImgs(self.ids[idx])[0]['file_name']))
        # print('Boxlist: ', target.bbox[0])
        # print('')
        # print('')

        idx = [self.coco.loadImgs(self.ids[idx])[0]['file_name'], self.coco.getAnnIds(imgIds=self.ids[idx]), self.ids[idx]] 

        return img, target, idx

    # pdb.set_trace()

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


#TODO:jinlong for three datasets
class COCODataset_da(torchvision.datasets.coco.CocoDetection):


    """
    ann_file_set: ann_file_source, ann_file_positive, ann_file_negative

    root_set: root_source, root_target_positive, root_target_negative 
    
    """

    def __init__(
        self, ann_file_set, root_set, remove_images_without_annotations, transforms=None, is_source= True
    ):
        super(COCODataset_da, self).__init__(annFile=ann_file_set[0], root=root_set[0])
        # sort indices for reproducible results

        # pdb.set_trace()

        self.root_source = root_set[0]
        self.root_target_positive = root_set[1]
        self.root_target_negative = root_set[2]

        self.coco_source = COCO(ann_file_set[0])
        self.coco_positive = COCO(ann_file_set[1])
        self.coco_negative = COCO(ann_file_set[2])

        self.coco_set = [self.coco_source, self.coco_positive, self.coco_negative]
        self.root_set = [self.root_source, self.root_target_positive, self.root_target_negative]


        self.ids = [sorted(list(self.coco_source.imgs.keys())), sorted(list(self.coco_positive.imgs.keys())), sorted(list(self.coco_negative.imgs.keys()))]

        # self.ids = sorted(self.ids)# for source dataset

        self.json_category_id_to_contiguous_id = [0,0,0]
        self.contiguous_category_id_to_json_id = [0,0,0]
        self.id_to_img_map = [0,0,0]

        # filter images without detection annotations
        for i in range(len(self.ids)):
            if remove_images_without_annotations:
                ids = []
                for img_id in self.ids[i]:
                    ann_ids = self.coco_set[i].getAnnIds(imgIds=img_id, iscrowd=None)
                    anno = self.coco_set[i].loadAnns(ann_ids)
                    if has_valid_annotation(anno):
                        ids.append(img_id)
                self.ids[i] = ids

            self.json_category_id_to_contiguous_id[i] = {
                v: i + 1 for i, v in enumerate(self.coco_set[i].getCatIds())
            }
            self.contiguous_category_id_to_json_id[i] = {
                v: k for k, v in self.json_category_id_to_contiguous_id[i].items()
            }
            self.id_to_img_map[i] = {k: v for k, v in enumerate(self.ids[i])}


        self.transforms = transforms
        self.is_source = is_source
        # pdb.set_trace()




    def __getitem__(self, idx):

        # img, anno = super(COCODataset_da, self).__getitem__(idx)


        # self.coco_set = [self.coco_source, self.coco_positive, self.coco_negative]
        # self.root_set = [self.root_source, self.root_target_positive, self.root_target_negative]


        img_id_set = [self.ids[0][idx],self.ids[1][idx], self.ids[2][idx]]

        ann_ids_set = [self.coco_set[0].getAnnIds(imgIds=img_id_set[0]),self.coco_set[1].getAnnIds(imgIds=img_id_set[1]), self.coco_set[2].getAnnIds(imgIds=img_id_set[2])]

        target_set = [self.coco_set[0].loadAnns(ann_ids_set[0]), self.coco_set[1].loadAnns(ann_ids_set[1]), self.coco_set[2].loadAnns(ann_ids_set[2])]


        path_set = [self.coco_set[0].loadImgs(img_id_set[0])[0]['file_name'], self.coco_set[1].loadImgs(img_id_set[1])[0]['file_name'], self.coco_set[2].loadImgs(img_id_set[2])[0]['file_name']]

        img_set = [0,0,0]

        final_img = [0,0,0]
        final_target = [0,0,0]

        for i in range(len(self.coco_set)):

            img = Image.open(os.path.join(self.root, path_set[i])).convert('RGB')
            img_set[i] = img


            # if self.transform is not None:
            #     img = self.transform(img)

            # if self.target_transform is not None:
            #     target = self.target_transform(target)



        for i in range(len(self.coco_set)):
            # filter crowd annotations
            # TODO might be better to add an extra field

            target_set[i] = [obj for obj in target_set[i] if obj["iscrowd"] == 0]

            boxes = [obj["bbox"] for obj in target_set[i]]
            boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
            target = BoxList(boxes, img_set[i].size, mode="xywh").convert("xyxy")

            classes = [obj["category_id"] for obj in target_set[i]]
            classes = [self.json_category_id_to_contiguous_id[i][c] for c in classes]
            classes = torch.tensor(classes)
            target.add_field("labels", classes)

            masks = [obj["segmentation"] for obj in target_set[i]]
            masks = SegmentationMask(masks, img_set[i].size)
            target.add_field("masks", masks)

            domain_labels = torch.ones_like(classes, dtype=torch.bool) if self.is_source else torch.zeros_like(classes, dtype=torch.bool)
            target.add_field("is_source", domain_labels)

            if target_set[i] and "keypoints" in target_set[i][0]:
                keypoints = [obj["keypoints"] for obj in target_set[i]]
                keypoints = PersonKeypoints(keypoints, img_set[i].size)
                target.add_field("keypoints", keypoints)

            target = target.clip_to_image(remove_empty=True)



            if self.transforms is not None:
                img_set[i], target = self.transforms(img_set[i], target)

            final_img[i] = img_set[i].clone()
            final_target[i] = target.clone()
        pdb.set_trace()
            
        return final_img, final_target, img_id_set

    # pdb.set_trace()

    def get_img_info(self, index):
        img_id = self.id_to_img_map[0][index]
        img_data = self.coco.imgs[img_id]
        return img_data