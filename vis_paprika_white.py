# -*- coding: utf-8 -*-
from glob import glob
import pandas as pd
import json
import cv2
import os
from skimage.color import rgb2lab
import numpy as np
import albumentations as albu
from PIL import ImageFont, ImageDraw, Image


train_files = sorted(glob('data/train/*'))
csv_files = sorted(glob(f'data/train/*/*.csv'))
test_files = sorted(glob('data/test/*'))
labelsss = pd.read_csv('data/train.csv')['label']

# Label Description : Refer to CSV File
crop = {'1': '딸기', '2': '토마토', '3': '파프리카', '4': '오이', '5': '고추', '6': '시설포도'}
disease = {'1': {'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '2': {'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)',
                 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'},
           '3': {'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '4': {'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '5': {'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍', 'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)',
                 'b8': '다량원소결핍 (K)'},
           '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}}
risk = {'1': '초기', '2': '중기', '3': '말기'}


label_description = {
# "1_00_0" : "딸기",
# "2_00_0" : "토마토",
# "2_a5_2" : "토마토_흰가루병_중기",
# "3_00_0" : "파프리카",
"3_a9_1" : "파프리카_흰가루병_초기",
# "3_a9_2" : "파프리카_흰가루병_중기",
# "3_a9_3" : "파프리카_흰가루병_말기",
# "3_b3_1" : "파프리카_칼슘결핍_초기",
# "3_b6_1" : "파프리카_다량원소결필(N)_초기",
# "3_b7_1" : "파프리카_다량원소결필(P)_초기",
# "3_b8_1" : "파프리카_다량원소결필(K)_초기",
# "4_00_0" : "오이",
# "5_00_0" : "고추",
#  "5_a7_2" : "고추_탄저병_중기",
#  "5_b6_1" : "고추_다량원소결필(N)_초기",
# "5_b7_1" : "고추_다량원소결필(P)_초기",
#  "5_b8_1" : "고추_다량원소결필(K)_초기",
# "6_00_0" : "시설포도",
# "6_a11_1" : "시설포도_탄저병_초기",
#  "6_a11_2" : "시설포도_탄저병_중기",
#  "6_a12_1" : "시설포도_노균병_초기",
# "6_a12_2" : "시설포도_노균병_중기",
#  "6_b4_1" : "시설포도_일소피해_초기",
#  "6_b4_3" : "시설포도_일소피해_말기",
# "6_b5_1" : "시설포도_축과병_초기"
}


label_encoder = {key: idx for idx, key in enumerate(label_description)}
label_decoder = {val: key for key, val in label_encoder.items()}



save_path = './vis_pap_white_1_GD'
os.makedirs(save_path,exist_ok=True)
num = 0
min_w = 0
max_w = 0
min_h =10000
max_h = 0
a_centercrop = albu.CenterCrop(384, 384, p=1)
a_resize = albu.Resize(384, 384,p=1)
a_randomresizecrop = albu.RandomResizedCrop(height=384, width=384,
                       scale=(0.45, 1.0), ratio=(0.75, 1.3333333333333333),
                       interpolation=1, p=1.0)

a_shiftscalerotate = albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                      rotate_limit=0, interpolation=1, border_mode=0,
                      value=0, p=1.0)
a_horizontalflip = albu.HorizontalFlip(p=1.0)
a_verticalflip = albu.VerticalFlip(p=1.0)
a_clahe = albu.CLAHE(clip_limit=2, p=1.0)
a_sharpen = albu.Sharpen(p=1.0)
a_randombrightnesscontrast = albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                              contrast_limit=(-0.1, 0.1), p=1.0)
a_rotate = albu.Rotate(border_mode=0, value=0, p=1.0)
a_saferotate = albu.SafeRotate(p=1.0)
a_randomrotate90 = albu.RandomRotate90(p=1.0)
a_perspective = albu.Perspective(p=1.0)
a_affine = albu.Affine(translate_percent=0.7, p=1.0)
a_randomcrop = albu.RandomCrop(height=384, width=384, p=1.0)
a_griddistortion = albu.GridDistortion(interpolation=cv2.INTER_NEAREST,border_mode=0,value=0,p=1.0)
base_transform_list = [
    albu.Resize(384, 384),
    albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                          rotate_limit=30, interpolation=1, border_mode=0,
                          value=0, p=0.5),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5)
]
transform = albu.Compose(
    base_transform_list +
    [
        albu.RandomRotate90(p=1.0),
        albu.CLAHE(clip_limit=2, p=0.25),
        albu.Sharpen(p=0.25),
        albu.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                      contrast_limit=(-0.1, 0.1), p=0.25),
        albu.RandomResizedCrop(height=384, width=384,
                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                               interpolation=1, p=1.0)

    ],bbox_params=albu.BboxParams(format='coco',label_fields=['class_labels']))
for file in train_files:
    file_name = file.split('/')[-1]

    json_path = f'{file}/{file_name}.json'
    image_path = f'{file}/{file_name}.jpg'
    with open(json_path, 'r') as f:
        json_file = json.load(f)

    crop = json_file['annotations']['crop']
    disease = json_file['annotations']['disease']
    risk = json_file['annotations']['risk']
    label = f'{crop}_{disease}_{risk}'
    parts = json_file['annotations']['part']
    bbox = json_file['annotations']['bbox'][0]

    if label in label_description:
        img = cv2.imread(image_path)
        for part in parts:
            x,y,w,h = int(part['x']),int(part['y']),int(part['w']),int(part['h'])
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)

        cv2.imwrite(f'{save_path}/{num:03d}_a.jpg', img)
        save_image_name = '_'.join(image_path.split('/'))

        h, w, _ = img.shape
        print(f'{file_name} | {h}  {w}')
        item = transform(image=img, bboxes=[[int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])]],class_labels=[label])


        # if w < 384:
        #     ratio = 384 / w
        #     new_h = int(h * ratio)
        #     img = cv2.resize(img, (384, new_h), interpolation=cv2.INTER_LINEAR)
        #     print(h,w,'->',new_h,384)
        # elif w == 384:
        #     pass
        # elif w>384 and w<=512:
        #     ratio = 384 / w
        #     new_h = int(h * ratio)
        #     img = cv2.resize(img, (384, new_h), interpolation=cv2.INTER_NEAREST_EXACT)
        #     print(h,w,'->',new_h,384)
        # else:
        #     ratio = 384 / h
        #     new_w = int(w * ratio)
        #     img = cv2.resize(img, (new_w, 384), interpolation=cv2.INTER_NEAREST_EXACT)
        #     print(h,w,'->',384, new_w)

        # img = a_centercrop(image=img)

        # img = a_resize(image=img)

        # img = a_randomrotate90(image=img)
        # img = img['image']
        #
        # img = a_randomcrop(image=img)
        # img = img['image']

        # img = a_randomrotate90(image=img)
        # img = img['image']

        # img = a_griddistortion(image=img)
        # img = img['image']
        # img = transform(image=img)
        img = item['image']

        bbox = item['bboxes'][0]
        x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
        # breakpoint()
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
        # img = a_rotate(image=img)
        # img = img['image']
        # img = a_saferotate(image=img)
        # img = img['image']


        # img = a_randomresizecrop(image=img)
        # img = img['image']
        # img = a_shiftscalerotate(image=img)
        # img = img['image']
        # img = a_horizontalflip(image=img)
        # img = img['image']
        # img = a_verticalflip(image=img)
        # img = img['image']
        # img = a_clahe(image=img)
        # img = img['image']
        # # img = a_sharpen(image=img)
        # # img = img['image']
        # img = a_randombrightnesscontrast(image=img)
        # img = img['image']
        #
        #

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./gulim.ttc", 20)
        org = (20, 20)
        text = label_description[label]
        draw.text(org, text, font=font, fill=(0, 0, 255))  # text를 출력 img = np.array(img) #다시 OpenCV가 처리가능하게 np 배열로 변환
        img = np.array(img)

        cv2.imwrite(f'{save_path}/{num:03d}_b.jpg', img)
        num+=1

        # cv2.imshow('',img)
        # if num>100:
        #     breakpoint()
        # print(img.shape)
        # if img.shape[1]<min_h:
        #     min_h = img.shape[1]
        # if img.shape[1]>max_h:
        #     max_h = img.shape[1]
        # num +=1
# print(num)
# print(min_h,max_h)
        # lab = rgb2lab(img)
        # # breakpoint()
        # img_clone = img.copy()



        # cv2.imwrite(f'{save_path}/{label}_{num:03d}_I.jpg', img_clone)

        # cv2.imwrite(f'{save_path}/{save_image_name}', img)
        # img_calhe = albu.CLAHE(clip_limit=2)(image=img)
        # img_sharpen = albu.Sharpen()(image=img)
        # img_emboss = albu.Emboss()(image=img)
        # img_random = albu.RandomBrightnessContrast()(image=img)
        # # breakpoint()
        # cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_0.jpg', img_calhe['image'])
        # cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_1.jpg', img_sharpen['image'])
        # cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_2.jpg', img_emboss['image'])
        # cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_3.jpg', img_random['image'])
        # # From Wikipedia "The CIELAB ... expresses color as three values: L* for the lightness from black (0) to white (100), a* from green (−) to red (+), and b* from blue (−) to yellow (+).
        # # L = lab[:, :, 0]
        # # white_mask_idx = L < 50
        # # L[white_mask_idx] = 0
        # # cv2.imwrite(f'{save_path}/{label}_{num:03d}_L.jpg', L)
        #
        # A = lab[:, :, 1]
        # # breakpoint()
        # green_mask_idx = A > 0
        # A[green_mask_idx] = 0
        # A = -A
        # cv2.imwrite(f'{save_path}/{label}_{num:03d}_LA.jpg', A)
        # # del label_description[label]
        # num +=1
        # if num > 100:
        #     break
