import os
import random
from typing import Any, Dict

import numpy as np
from mmcls.models.losses import accuracy, f1_score, precision, recall
from mmcls.models.losses.eval_metrics import class_accuracy

from .base_dataset import BaseDataset
from .builder import DATASETS

from .fertools import FER_CLASSES, find_folders, get_samples, convert2coarse_label


@DATASETS.register_module()
class RAF(BaseDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    DATASET_CLASSES = [
        'Surprise',
        'Fear',
        'Disgust',
        'Happiness',
        'Sadness',
        'Anger',
        'Neutral'
    ]
    CLASSES = FER_CLASSES[:7]

    @staticmethod
    def convert_gt_label(i:int):
        """# dataset -> FER_CLASSES"""
        convert_table = (5, 2, 1, 4, 3, 0, 6)
        assert sum(convert_table) == sum([i for i in range(7)])
        return convert_table[i]

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
            samples = [[i[0].replace('_aligned', ''), i[1]] for i in samples]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            gt_label = int(gt_label) - 1
            gt_label = self.convert_gt_label(gt_label)
            coarse_label = convert2coarse_label(gt_label)
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['coarse_label'] = np.array(coarse_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

