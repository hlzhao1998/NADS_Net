import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import numpy as np
from pycocotools.coco import COCO

from entity import params


class CocoDataLoader(object):
    def __init__(self, coco, mode='train'):
        self.coco = coco
        assert mode in ['train', 'val'], 'Data loading mode is invalid.'
        self.mode = mode
        self.catIds = coco.getCatIds()  # catNms=['person']
        self.imgIds = sorted(coco.getImgIds(catIds=self.catIds))

    def __len__(self):
        return len(self.imgIds)

    def gen_masks(self, img, annotations):
        mask_all = np.zeros(img.shape[:2], 'bool')
        mask_miss = np.zeros(img.shape[:2], 'bool')
        for ann in annotations:
            mask = self.coco.annToMask(ann).astype('bool')
            if ann['iscrowd'] == 1:
                intxn = mask_all & mask
                mask_miss = np.bitwise_or(mask_miss.astype(int) , np.subtract(mask, intxn, dtype=np.int32))
                mask_all = np.bitwise_or(mask_all.astype(int) , mask.astype(int))
            elif ann['num_keypoints'] < params['min_keypoints'] or ann['area'] <= params['min_area']:
                mask_all = np.bitwise_or(mask_all.astype(int) , mask.astype(int))
                mask_miss = np.bitwise_or(mask_miss.astype(int) , mask.astype(int))
            else:
                mask_all = np.bitwise_or(mask_all.astype(int) , mask.astype(int))
        return mask_all, mask_miss

    def dwaw_gen_masks(self, img, mask, color=(0, 0, 1)):
        bimsk = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * bimsk.astype(np.int32)
        clmsk = np.ones(bimsk.shape) * bimsk
        for i in range(3):
            clmsk[:, :, i] = clmsk[:, :, i] * color[i] * 255
        img = img + 0.7 * clmsk - 0.7 * mskd
        return img.astype(np.uint8)

    def draw_masks_and_keypoints(self, img, annotations):
        for ann in annotations:
            # masks
            mask = self.coco.annToMask(ann).astype(np.uint8)
            if ann['iscrowd'] == 1:
                color = (0, 0, 1)
            elif ann['num_keypoints'] == 0:
                color = (0, 1, 0)
            else:
                color = (1, 0, 0)
            bimsk = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mskd = img * bimsk.astype(np.int32)
            clmsk = np.ones(bimsk.shape) * bimsk
            for i in range(3):
                clmsk[:, :, i] = clmsk[:, :, i] * color[i] * 255
            img = img + 0.7 * clmsk - 0.7 * mskd

            # keypoints
            for x, y, v in np.array(ann['keypoints']).reshape(-1, 3):
                if v == 1:
                    cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
                elif v == 2:
                    cv2.circle(img, (x, y), 3, (255, 0, 255), -1)
        return img.astype(np.uint8)

    def get_img_annotation(self, ind=None, img_id=None):
        """インデックスまたは img_id から coco annotation dataを抽出、条件に満たない場合はNoneを返す """
        if ind is not None:
            img_id = self.imgIds[ind]

        anno_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(anno_ids)

        img_file = os.path.join(params['coco_dir'], self.mode+'2017', self.coco.loadImgs([img_id])[0]['file_name'])
        img = cv2.imread(img_file)
        return img, annotations, img_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true', help='visualize annotations and ignore masks')
    args = parser.parse_args()

    for mode in ['train', 'val']:
        coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_{}2017.json'.format(mode)))
        data_loader = CocoDataLoader(coco, mode=mode)

        save_dir = os.path.join(params['coco_dir'], 'ignore_mask_{}2017'.format(mode))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in tqdm(range(len(data_loader))):
            img, annotations, img_id = data_loader.get_img_annotation(ind=i)
            mask = np.zeros(img.shape[:2]).astype(np.uint8)
            for ann in annotations:
                mask_ = coco.annToMask(ann).astype(np.uint8)
                mask = np.bitwise_or(mask_.astype(np.int), mask.astype(np.int))
            if np.any(mask):
                mask = mask.astype(np.uint8) * 255
                save_path = os.path.join(save_dir, '{:012d}.png'.format(img_id))
                cv2.imwrite(save_path, mask)
            if args.vis:
                cv2.imshow('mask', mask)
                cv2.waitKey(1)