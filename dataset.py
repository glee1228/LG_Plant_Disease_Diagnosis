import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

import json
import torch
from torch.utils.data import Dataset
import albumentations as albu
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from albumentations.pytorch import ToTensorV2

class PlantDACON(Dataset):
    def __init__(self,
                 image_size,
                 files,
                 csv_files=None,
                 avail_features=None,
                 label_description=None,
                 aug_ver=0,
                 tta_transform=None,
                 mode='train'):
        self.mode = mode
        self.image_size = image_size
        self.files = files
        self.aug_ver = aug_ver
        self.avail_features = avail_features
        self.csv_files = csv_files
        self.csv_feature_check = [0] * len(self.files)
        self.csv_features = [None] * len(self.files)
        self.max_len = 320
        self.label_description = label_description
        self.label_encoder = {key: idx for idx, key in enumerate(self.label_description)}
        self.base_transform_list = [
            albu.Resize(self.image_size, self.image_size),
            albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                  rotate_limit=30, interpolation=1, border_mode=0,
                                  value=0, p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5)
        ]


        self.tta_transform = tta_transform
        self.transform_normalize = [
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2()
        ]
        if self.mode == 'train':
            # Only-flip
            if self.aug_ver == 1:
                self.transform = albu.Compose([albu.Resize(self.image_size, self.image_size),
                                               albu.HorizontalFlip(p=0.5),
                                               albu.VerticalFlip(p=0.5),
                                               albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                               ToTensorV2(),
                                               ])
            # Base Augmentation
            elif self.aug_ver == 2:
                self.transform = albu.Compose(
                    self.base_transform_list
                    + self.transform_normalize)

            # Base + Transpose
            elif self.aug_ver == 3:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                    ]
                    + self.transform_normalize)

            # Base + HueSaturationValue
            elif self.aug_ver == 4:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    ]
                    + self.transform_normalize)

            # Base + RandomBrightnessContrast
            elif self.aug_ver == 5:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                    ]
                    + self.transform_normalize)

            # Base + CLAHE
            elif self.aug_ver == 6:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.CLAHE(p=0.5),
                    ]
                    + self.transform_normalize)

            # Base + Sharpen
            elif self.aug_ver == 7:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.CLAHE(p=0.5),
                        albu.Sharpen(p=0.5),
                    ]
                    + self.transform_normalize)

            # Base + Transpose + HueSaturationValue
            elif self.aug_ver == 8:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    ]
                    + self.transform_normalize)

            # Base + Transpose + RandomBrightnessContrast
            elif self.aug_ver == 9:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),

                    ]
                    + self.transform_normalize)
            # Base + Transpose + CLAHE
            elif self.aug_ver == 10:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.CLAHE(p=0.5),

                    ]
                    + self.transform_normalize)
            # Base + Transpose + Sharpen
            elif self.aug_ver == 11:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.Sharpen(p=0.5),

                    ]
                    + self.transform_normalize)

            # Base + Transpose + HueSaturationValue + RandomBrightnessContrast
            elif self.aug_ver == 12:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),

                    ]
                    + self.transform_normalize)
            # Base + Transpose + HueSaturationValue + CLAHE
            elif self.aug_ver == 13:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                        albu.CLAHE(p=0.5),

                    ]
                    + self.transform_normalize)
            # Base + Transpose + HueSaturationValue + Sharpen
            elif self.aug_ver == 14:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                        albu.Sharpen(p=0.5),

                    ]
                    + self.transform_normalize)
            # Base + Transpose + HueSaturationValue + Sharpen
            elif self.aug_ver == 15:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                        albu.CLAHE(p=0.5),
                    ]
                    + self.transform_normalize)

            # Base + Transpose + HueSaturationValue + Sharpen
            elif self.aug_ver == 16:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                        albu.Sharpen(p=0.5),
                    ]
                    + self.transform_normalize)
            # Base + Transpose + HueSaturationValue + Sharpen
            elif self.aug_ver == 17:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.CLAHE(p=0.5),
                        albu.Sharpen(p=0.5),
                    ]
                    + self.transform_normalize)
            # Base + Transpose + HueSaturationValue + Sharpen
            elif self.aug_ver == 18:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                        albu.CLAHE(p=0.5),
                    ]
                    + self.transform_normalize)
            # Base + Transpose + HueSaturationValue + Sharpen
            elif self.aug_ver == 19:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                        albu.Sharpen(p=0.5),
                    ]
                    + self.transform_normalize)
            # Base + Transpose + HueSaturationValue + Sharpen
            elif self.aug_ver == 20:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.5),
                        albu.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                        albu.CLAHE(p=0.5),
                        albu.Sharpen(p=0.5),
                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 21:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.5),
                        albu.CLAHE(p=0.5),
                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 22:
                self.transform = albu.Compose(
                    [
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),
                        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                              rotate_limit=30, interpolation=1, border_mode=0,
                                              value=0, p=0.5),
                        albu.HorizontalFlip(p=0.5),
                        albu.VerticalFlip(p=0.5),
                        albu.CLAHE(clip_limit=2,p=0.25),
                        albu.Sharpen(p=0.25),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.25),

                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 23:
                self.transform = albu.Compose(
                    [
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),
                        albu.RandomRotate90(p=1.0),
                        albu.HorizontalFlip(p=0.5),
                        albu.VerticalFlip(p=0.5),
                        albu.CLAHE(clip_limit=2,p=0.25),
                        albu.Sharpen(p=0.25),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.25),

                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 24:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.CLAHE(clip_limit=2, p=0.25),
                        albu.Sharpen(p=0.25),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.25),
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),
                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 25:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.Transpose(p=0.25),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.25),
                        albu.Sharpen(p=0.25),
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),
                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 26:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.25),
                        albu.Sharpen(p=0.25),
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),
                    ]
                    + self.transform_normalize)

            elif self.aug_ver == 27:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.RandomRotate90(p=0.5),
                        albu.OneOf(
                            [
                                albu.CLAHE(clip_limit=2, p=0.5),
                                albu.Sharpen(p=0.5),
                                albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                  contrast_limit=(-0.1, 0.1), p=0.5)],
                            p=1.0),
                        albu.GridDistortion(interpolation=1,
                                            border_mode=0,
                                            value=0,
                                            p=0.25),
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),


                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 28:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.CLAHE(clip_limit=2, p=0.25),
                        albu.Sharpen(p=0.25),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.25),
                        albu.GridDistortion(interpolation=1,
                                            border_mode=0,
                                            value=0,
                                            p=0.25),
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),
                    ]
                    + self.transform_normalize)
            elif self.aug_ver == 29:
                self.transform = albu.Compose(
                    self.base_transform_list +
                    [
                        albu.RandomRotate90(p=1.0),
                        albu.CLAHE(clip_limit=2, p=0.25),
                        albu.Sharpen(p=0.25),
                        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                      contrast_limit=(-0.1, 0.1), p=0.25),
                        albu.RandomResizedCrop(height=self.image_size, width=self.image_size,
                                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                               interpolation=1, p=1.0),
                    ]
                    + self.transform_normalize)
            else :
                self.transform = albu.Compose([albu.Resize(self.image_size, self.image_size),
                                               albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                               ToTensorV2(),
                                               ])
        elif self.mode == 'valid':
            # self.transform = albu.Compose([
            #     albu.CenterCrop(self.image_size, self.image_size),
            #     albu.HorizontalFlip(p=0.5),
            #     albu.VerticalFlip(p=0.5),
            #     albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            #     ToTensorV2(),
            # ])
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])
        elif self.mode == 'test':
            if self.tta_transform:
                self.transform = albu.Compose(
                    [albu.Resize(self.image_size, self.image_size)]+
                    [
                        self.tta_transform
                    ]
                    + self.transform_normalize)
            else:
                # self.transform = albu.Compose([
                #     albu.CenterCrop(self.image_size, self.image_size),
                #     albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                #     ToTensorV2(),
                # ])
                self.transform = albu.Compose([
                    albu.Resize(self.image_size, self.image_size),
                    albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                    ToTensorV2(),
                ])
        else:
            self.transform = albu.Compose([
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ])


        if avail_features:
            # self.csv_feature_dict = self.make_csv_feature_dict()
            self.csv_feature_dict = {'내부 온도 1 평균': [3.4, 47.3],
                                     '내부 온도 1 최고': [3.4, 47.6],
                                     '내부 온도 1 최저': [3.3, 47.0],
                                     '내부 습도 1 평균': [23.7, 100.0],
                                     '내부 습도 1 최고': [25.9, 100.0],
                                     '내부 습도 1 최저': [0.0, 100.0],
                                     '내부 이슬점 평균': [0.1, 34.5],
                                     '내부 이슬점 최고': [0.2, 34.7],
                                     '내부 이슬점 최저': [0.0, 34.4]}
    def make_csv_feature_dict(self):
        # 분석에 사용할 feature 선택
        temp_csv = pd.read_csv(self.csv_files[0])[self.avail_features]
        max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

        # feature 별 최대값, 최솟값 계산
        for csv in tqdm(self.csv_files[1:]):
            temp_csv = pd.read_csv(csv)[self.avail_features]
            temp_csv = temp_csv.replace('-', np.nan).dropna()
            if len(temp_csv) == 0:
                continue
            temp_csv = temp_csv.astype(float)
            temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
            max_arr = np.max([max_arr, temp_max], axis=0)
            min_arr = np.min([min_arr, temp_min], axis=0)

        # feature 별 최대값, 최솟값 dictionary 생성
        csv_feature_dict = {self.avail_features[i]: [min_arr[i], max_arr[i]] for i in range(len(self.avail_features))}
        return csv_feature_dict

    def get_csv_feature_dict(self):
        return self.csv_feature_dict

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]

        json_path = f'{file}/{file_name}.json'
        image_path = f'{file}/{file_name}.jpg'
        if self.avail_features:
            if self.csv_feature_check[i] == 0:
                csv_path = f'{file}/{file_name}.csv'
                df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
                df = df.replace('-', 0)
                # MinMax scaling
                for col in df.columns:
                    df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                    df[col] = df[col] / (self.csv_feature_dict[col][1] - self.csv_feature_dict[col][0])
                # zero padding
                pad = np.zeros((self.max_len, len(df.columns)))
                length = min(self.max_len, len(df))
                pad[-length:] = df.to_numpy()[-length:]
                # transpose to sequential data
                csv_feature = pad.T
                self.csv_features[i] = csv_feature
                self.csv_feature_check[i] = 1
            else:
                csv_feature = self.csv_features[i]
        else:
            pass

        img = cv2.imread(image_path)
        # if self.mode in ['valid','test']:
        #     h, w, _ = img.shape
        #     if w < 384:
        #         ratio = 384 / w
        #         new_h = int(h * ratio)
        #         img = cv2.resize(img, (384, new_h), interpolation=cv2.INTER_LINEAR)
        #         # print(h, w, '->', new_h, 384)
        #     elif w == 384:
        #         pass
        #     elif w > 384 and w <= 512:
        #         ratio = 384 / w
        #         new_h = int(h * ratio)
        #         img = cv2.resize(img, (384, new_h), interpolation=cv2.INTER_NEAREST_EXACT)
        #         # print(h, w, '->', new_h, 384)
        #     else:
        #         ratio = 384 / h
        #         new_w = int(w * ratio)
        #         img = cv2.resize(img, (new_w, 384), interpolation=cv2.INTER_NEAREST_EXACT)
        #         # print(h, w, '->', 384, new_w)

        img = self.transform(image=img)

        if self.avail_features:
            if self.mode in ['train', 'valid']:
                with open(json_path, 'r') as f:
                    json_file = json.load(f)

                crop = json_file['annotations']['crop']
                disease = json_file['annotations']['disease']
                risk = json_file['annotations']['risk']
                label = f'{crop}_{disease}_{risk}'
                return {
                    'img': img,
                    'csv_feature': torch.tensor(csv_feature, dtype=torch.float32),
                    'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
                }

            else:
                return {
                    'img': img,
                    'csv_feature': torch.tensor(csv_feature, dtype=torch.float32)
                }
        else:
            if self.mode in ['train', 'valid']:
                with open(json_path, 'r') as f:
                    json_file = json.load(f)

                crop = json_file['annotations']['crop']
                disease = json_file['annotations']['disease']
                risk = json_file['annotations']['risk']
                label = f'{crop}_{disease}_{risk}'

                return {
                    'img': img,
                    'label': torch.tensor(self.label_encoder[label], dtype=torch.long)
                }

            else:
                return {
                    'img': img,
                }

