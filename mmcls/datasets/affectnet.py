import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS

from .fertools import FER_CLASSES, find_folders, get_samples, convert2coarse_label


@DATASETS.register_module()
class AffectNet(BaseDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    DATASET_CLASSES = [
        'Neutral',  # 0
        'Happy',    # 1
        'Sad',      # 2
        'Surprise', # 3
        'Fear',     # 4
        'Disgust',  # 5
        'Anger',    # 6
        'Contempt'  # 7
    ]

    def __init__(self, data_prefix, pipeline, classes=None, ann_file=None, test_mode=False, num_classes=7):
        self.num_classes = num_classes

        if not classes:
            self.CLASSES = FER_CLASSES[:num_classes]
        else:
            self.CLASSES = classes

        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=self.CLASSES,
            ann_file=ann_file,
            test_mode=test_mode
        )


    @staticmethod
    def convert_gt_label(i:int, num_classes=7):
        """# dataset -> FER_CLASSES"""
        convert_table = (6, 5, 4, 2, 1, 3, 0, 7)
        assert sum(convert_table[:num_classes]) == sum([i for i in range(num_classes)])
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
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            gt_label = int(gt_label) # - 1 AffectNet class indices are 0-based
            gt_label = self.convert_gt_label(gt_label, self.num_classes)
            coarse_label = convert2coarse_label(gt_label)
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['coarse_label'] = np.array(coarse_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
