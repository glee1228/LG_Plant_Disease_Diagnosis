# -*- coding: utf-8 -*-
from glob import glob
import pandas as pd
import json
import cv2
import os
from skimage.color import rgb2lab
import numpy as np
import albumentations as albu

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
"3_a9_2" : "파프리카_흰가루병_중기",
"3_a9_3" : "파프리카_흰가루병_말기",
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



save_path = './vis'
os.makedirs(save_path,exist_ok=True)
num = 0
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
    if label in label_description:
        img = cv2.imread(image_path)
        lab = rgb2lab(img)
        # breakpoint()
        img_clone = img.copy()
        for part in parts:
            x,y,w,h = int(part['x']),int(part['y']),int(part['w']),int(part['h'])
            img_clone = cv2.rectangle(img_clone, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.imwrite(f'{save_path}/{label}_{num:03d}_I.jpg', img_clone)

        img_calhe = albu.CLAHE(clip_limit=2)(image=img)
        img_sharpen = albu.Sharpen()(image=img)
        img_emboss = albu.Emboss()(image=img)
        img_random = albu.RandomBrightnessContrast()(image=img)
        # breakpoint()
        cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_0.jpg', img_calhe['image'])
        cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_1.jpg', img_sharpen['image'])
        cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_2.jpg', img_emboss['image'])
        cv2.imwrite(f'{save_path}/{label}_{num:03d}_I_3.jpg', img_random['image'])
        # From Wikipedia "The CIELAB ... expresses color as three values: L* for the lightness from black (0) to white (100), a* from green (−) to red (+), and b* from blue (−) to yellow (+).
        # L = lab[:, :, 0]
        # white_mask_idx = L < 50
        # L[white_mask_idx] = 0
        # cv2.imwrite(f'{save_path}/{label}_{num:03d}_L.jpg', L)

        A = lab[:, :, 1]
        # breakpoint()
        green_mask_idx = A > 0
        A[green_mask_idx] = 0
        A = -A
        cv2.imwrite(f'{save_path}/{label}_{num:03d}_LA.jpg', A)
        # del label_description[label]
        num +=1
        if num > 100:
            break
