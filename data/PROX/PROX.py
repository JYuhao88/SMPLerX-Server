import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from main.config import cfg
from common.utils.human_models import smpl_x
from common.utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from common.utils.transforms import world2cam, cam2pixel, rigid_align
from humandata import HumanDataset


class PROX(HumanDataset):
    def __init__(self, transform, data_split):
        super(PROX, self).__init__(transform, data_split)

        if self.data_split == 'train':
            filename = getattr(cfg, 'filename', 'prox_train_smplx_new.npz')
        else:
            raise ValueError('PROX test set is not support')

        self.img_dir = osp.join(cfg.data_dir, 'PROX')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', filename)
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {}

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)