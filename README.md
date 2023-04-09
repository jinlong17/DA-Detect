<!--
 * @Descripttion: 
 * @version: 
 * @Author: Jinlong Li CSU PhD
 * @Date: 2021-10-15 17:13:40
 * @LastEditors: Jinlong Li CSU PhD
 * @LastEditTime: 2023-04-09 18:08:34
-->
# [DA-Detect](https://arxiv.org/abs/2210.15176): Domain Adaptive Object Detection for Autonomous Driving under Foggy Weather (WACV 2023)



[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2210.15176)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
<!-- [![supplement](https://img.shields.io/badge/Supplementary-Material-red)]() -->
<!-- [![video](https://img.shields.io/badge/Video-Presentation-F9D371)]() -->


This is a PyTorch implementation of 'Domain Adaptive Object Detection for Autonomous Driving under Foggy Weather', implemented by [jinlong Li](https://jinlong17.github.io/).

![teaser](image/DA_faster_rcnn.png)

## Installation

Please follow the instruction in [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) to install it, the detail as follows:

* Clone this repository on your PC;
  ```bash
  git clone https://github.com/jinlong17/DA-Detect
  ```
* Come into the cloned repository, the following the [INSTALLATION](https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/INSTALL.md), like (Linux as an example):
    ```bash
    # first, make sure that your conda is setup properly with the right environment for that, check that `which conda`, `which pip` and `which python` points to the right path. From a clean conda env, this is what you need to do
    conda create --name maskrcnn_benchmark -y
    conda activate maskrcnn_benchmark
    # this installs the right pip and dependencies for the fresh python
    conda install ipython pip
    # maskrcnn_benchmark and coco api dependencies
    pip install ninja yacs cython matplotlib tqdm opencv-python
    # follow PyTorch installation in https://pytorch.org/get-started/locally/
    # we give the instructions for CUDA 9.0
    conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0
    export INSTALL_DIR=$PWD
    # install pycocotools
    cd $INSTALL_DIR
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext install
    # install cityscapesScripts
    cd $INSTALL_DIR
    git clone https://github.com/mcordts/cityscapesScripts.git
    cd cityscapesScripts/
    python setup.py build_ext install
    # install apex
    cd $INSTALL_DIR
    git clone https://github.com/NVIDIA/apex.git
    cd apex
    python setup.py install --cuda_ext --cpp_ext
    # install PyTorch Detection
    cd $INSTALL_DIR
    git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
    cd maskrcnn-benchmark
    # the following will install the lib with
    # symbolic links, so that you can modify
    # the files if you want and won't need to
    # re-build it
    python setup.py build develop
    unset INSTALL_DIR
    # or if you are on macOS
    # MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
    ```
---
* If you meet the error ``` RuntimeError: Error compiling objects for extension ``` when running```python setup.py build develop ```, you can use commands as follows, which is in the [link](https://github.com/amazon-science/siam-mot/blob/main/readme/INSTALL.md) :
```bash
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
# You can then run the regular setup command
python3 setup.py build develop
```
* If you meet the eoor ```UnicodeDecodeError: 'ascii' codec can't decode byte 0xe9 in position  ``` when you are training, you can solve it by modifying that

```python
if torch._six.PY37:
    data = pickle.load(f, encoding="latin1")
else:
    # data = pickle.load(f)
    data = pickle.load(f, encoding="latin1")
```
* If you meet the warning ```Warning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead ``` or ``` IndexError: list index out of range ``` when you are training, you can solve it following the [solution](https://github.com/facebookresearch/maskrcnn-benchmark/issues/1182) by substitue for all ```torch.bool``` with ```torch.uint8```;

## DATA

1. Download the dataset;
     * **Source domain**:  leftImg8bit_trainvaltest in [Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/)
     * **Target domain**: leftImg8bit_trainvaltest_foggy in [Foggy Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/);
     * **Auxiliary domain**: for a domain-level metric regularization (use for Triplet loss):  
       * Download the [rainy mask](https://github.com/tsingqguo/efficientderain) (rainmix/Streaks_Garg06.zip); 
       * Set your paths for rainy mask and Cityscape dataset in the code efficentderain-master/generate_rainy_cityscape.py, then to generate the rain cityscape dataset.
2. Follow the example in [Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN) to generate coco style annoation files (**Cityscapes Dataset** and **Foggy Cityscapes Dataset**)


## Getting Started
An example of Domain Adaptive Faster R-CNN using triplet loss with ResNet adapting from **Cityscapes** dataset to **Foggy Cityscapes** dataset is provided:
1. Follow the example in [Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN) to generate coco style annoation files
2. Modified the dataset path to the Cityscapes and Foggy Cityscapes dataset and Rainy Cityscapes dataset in `paths_catalog.py`, which is like:
    ```bash
    "foggy_cityscapes_fine_instanceonly_seg_train_cocostyle":{
        "img_dir": "your data path",
        "ann_file": "your data path" 
    }
    "foggy_cityscapes_fine_instanceonly_seg_val_cocostyle":{
        "img_dir": "your data path",
        "ann_file": "your data  path" 
    }
    ```
3. Modified the yaml file in `configs/da_faster_rcnn`, use `e2e_triplet_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml` as an example:
   ```bash
    
    MODEL:
        OUTPUT_DIR: #your training model saving path
        OUTPUT_SAVE_NAME: # your training model saving folder name
    DA_HEADS:
        DA_ADV_GRL: # True or False, using AdvGRL or GRL
        ALIGNMENT: # True or False,  True is for aligned synthetic dataset training like: Cityscapes dataset, Foggy Cityscapes dataset, and Rainy Cityscapes dataset. False is for cross-camera training.
        TRIPLET_MARGIN: # the margin of triplet loss 
    ```
4. Train the Domain Adaptive Faster R-CNN:
    ```
    python3 tools/train_net.py --config-file "configs/da_faster_rcnn/e2e_triplet_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml"
    ```
5. Test the trained model:
    ```
    python3 tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_triplet_da_faster_rcnn_R_50_C4_cityscapes_to_foggy_cityscapes.yaml" MODEL.WEIGHT <path_to_store_weight>/model_final.pth
    ```

## Proposed Component

### Adversarial Gradient Reversal Layer (**AdvGRL**)
Illustration of the adversarial mining for hard training examples by the proposed AdvGRL. In this example, we set $\lambda_0$ = 1, $\beta$ = 30. Harder training examples with lower domain classifier loss $L_c$ will have larger response. The function `Adv_GRL()` can be in file `modeling/da_heads.py`. 
![teaser](image/grl_loss.png)


### Domain-level Metric Regularization (Based on Triplet Loss)

Previous existing domain adaptation methods mainly focus on the transfer learning from source domain $S$ to target domain $T$, which neglects the potential benefits of the third related domain can bring. To address this and thus additionally involve the feature metric constraint between different domains, we proposed an auxiliary domain for a domain-level metric regularization during the transfer learning. The function `Domainlevel_Img_component()`and `Domainlevel_Ins_component()` can be found in file `modeling/da_heads.py`.


## Ablation Study Results
The following results are conducted with the same RestNet-50 backbone on the Cityscapes -> Foggy Cityscapes experiment.

|                                | Image-level  | Object-level    |    AdvGRL    | Regularization | AP@50       | Download |
|--------------------------------|:------------:|:---------------:|:------------:|:--------------:|:-----------:|:--------:|
| Faster R-CNN (source only)     |              |                 |              |                |   23.41     |          |         
| DA Faster (Img+GRL)            |          ✓   |                 |              |                |   38.10     |          | 
| DA Faster (Obj+GRL)            |              |          ✓      |              |                |   38.02     |          |
| DA Faster (Img+Obj+GRL)        |          ✓   |          ✓      |              |                |   38.43     |          | 
| DA Faster (Img+Obj+AdvGRL)     |          ✓   |          ✓      |          ✓   |                |   40.23     |          |
| DA Faster (Img+Obj+GRL+Reg)    |          ✓   |          ✓      |              |        ✓       |   41.97     |          |
| DA Faster (Img+Obj+AdvGRL+Reg) |          ✓   |          ✓      |          ✓   |        ✓       |   42.34     |          |



## Citation
 If you are using our proposed method for your research, please cite the following paper:
 ```bibtex
@inproceedings{li2023domain,
  title={Domain Adaptive Object Detection for Autonomous Driving under Foggy Weather},
  author={Li, Jinlong and Xu, Runsheng and Ma, Jin and Zou, Qin and Ma, Jiaqi and Yu, Hongkai},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={612--622},
  year={2023}
}
```

## Acknowledgment
 This code is modified based on the code [Domain-Adaptive-Faster-RCNN-PyTorch](https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Thanks.




