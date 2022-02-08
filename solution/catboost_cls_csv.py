import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import cv2
import matplotlib.pyplot as plt

import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score
import time

def fix_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def accuracy_function(real, pred):
    score = f1_score(real, pred, average='macro')
    return score

def model_save(model, score,  path):
    os.makedirs('model', exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'score': score
    }, path)


device = torch.device('cuda')
fix_everything(1111)

train_csv = sorted(glob('data/train/*/*.csv'))
train_jpg = sorted(glob('data/train/*/*.jpg'))
train_json = sorted(glob('data/train/*/*.json'))

crops = []
diseases = []
risks = []
labels = []

for i in range(len(train_json)):
    with open(train_json[i], 'r') as f:
        sample = json.load(f)
        crop = sample['annotations']['crop']
        disease = sample['annotations']['disease']
        risk = sample['annotations']['risk']

        # Make Labels
        label = f"{crop}_{disease}_{risk}"

        crops.append(crop)
        diseases.append(disease)
        risks.append(risk)
        labels.append(label)

label_unique = sorted(np.unique(labels))
label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}

labels = [label_unique[k] for k in labels]
# breakpoint()
# 분석에 사용할 feature 선택
# csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
#                 '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저', '내부 CO2 평균',
#                 '내부 CO2 최고', '내부 CO2 최저']

csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
                '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

csv_files = sorted(glob('data/train/*/*.csv'))
max_arr, min_arr = [-1e9] * 9, [1e9] * 9

# feature 별 최대값, 최솟값 계산
# for csv in tqdm(csv_files[0:]):
#     temp_csv = pd.read_csv(csv)[csv_features]
#     temp_csv = temp_csv.replace('-', np.nan).dropna()
#     if len(temp_csv) == 0:
#         continue
#     temp_csv = temp_csv.astype(float)
#     temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
#     max_arr = np.max([max_arr, temp_max], axis=0)
#     min_arr = np.min([min_arr, temp_min], axis=0)
#
# # feature 별 최대값, 최솟값 dictionary 생성
# csv_feature_dict = {csv_features[i]: [min_arr[i], max_arr[i]] for i in range(len(csv_features))}
imgs = train_jpg

csv_feature_dict = {'내부 온도 1 평균': [3.4, 47.3],
                         '내부 온도 1 최고': [3.4, 47.6],
                         '내부 온도 1 최저': [3.3, 47.0],
                         '내부 습도 1 평균': [23.7, 100.0],
                         '내부 습도 1 최고': [25.9, 100.0],
                         '내부 습도 1 최저': [0.0, 100.0],
                         '내부 이슬점 평균': [0.1, 34.5],
                         '내부 이슬점 최고': [0.2, 34.7],
                         '내부 이슬점 최저': [0.0, 34.4]}

import numpy as np
from torchvision import transforms


class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, mode='train'):

        self.img_paths = img_paths
        self.labels = labels
        self.mode = mode
        self.transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=15, scale=[1.2, 1.2]),
            # transforms.ColorJitter(
            #     brightness=0.4,
            #     contrast=0.4,
            #     saturation=0.4,
            # ),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # FOR CSV
        self.max_len = 320
        self.csv_feature_dict = csv_feature_dict
        self.csv_feature_check = [0] * len(self.img_paths)
        self.csv_features = [None] * len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        csv_path = img.replace("jpg", "csv")
        img = Image.open(img)

        # GET CSV FEATURE
        if self.csv_feature_check[idx] == 0:

            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)

            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col][df[col] < 0] = 0
                df[col] = df[col] / (self.csv_feature_dict[col][1] - self.csv_feature_dict[col][0])

            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]

            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[idx] = csv_feature
            self.csv_feature_check[idx] = 1

        else:
            csv_feature = self.csv_features[idx]

        if self.mode == 'test':
            return self.test_transforms(img), csv_feature

        # Heavy Augmentation
        if self.mode != 'test':
            img = self.transforms(img)

        label = self.labels[idx]
        return img, csv_feature, label

folds = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, valid_idx in kf.split(imgs, labels):
    folds.append((train_idx, valid_idx))


fold= 0
batch_size = 16
epochs = 15
train_idx = folds[fold][0]
valid_idx = folds[fold][1]
train_dataset = Custom_dataset(np.array(imgs)[train_idx], np.array(labels)[train_idx], mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=8)

valid_dataset = Custom_dataset(np.array(imgs)[valid_idx], np.array(labels)[valid_idx], mode='valid')
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=8)

# For Machine Learning
train_x = []
train_y = []
train_imgs = []


from baseline import mixup_criterion, mixup_data
# batch_iter = tqdm(enumerate(train_loader), 'Training', total=len(train_loader), ncols=150)

# num = 0
# save_path = './mixed_data'
# os.makedirs(save_path,exist_ok=True)
# for batch_idx, batch_item in batch_iter:
#     img, csv_feature, label = batch_item
#
#     if batch_idx % 4 == 0:  # 4 step 주기로 mixup 사용
#         mixed_x, y_a, y_b, lam = mixup_data(img, label)
#         mixed_x = np.array(mixed_x)
        # breakpoint()
        # for m_i in range(len(mixed_x)):
        #     cv2.imwrite(f'{save_path}/{num:03d}_a.jpg', np.transpose(mixed_x[m_i]*255.,(2,1,0)))
        #     num+=1
        # breakpoint()

for img, csv_feature, label in tqdm(train_dataset):
    # if label in [4,5,6]:
    if label in [4, 5]:
        if label==4:
            label=0
        elif label==5:
            label=1
        # elif label==6:
        #     label=2
    train_imgs.append(img)
    train_x.append(csv_feature)
    train_y.append(label)

valid_x = []
valid_y = []
valid_imgs = []

for img, csv_feature, label in tqdm(valid_dataset):
    # if label in [4, 5, 6]:
    if label in [4, 5]:
        if label == 4:
            label = 0
        elif label == 5:
            label = 1
        # elif label == 6:
        #     label = 2
    valid_imgs.append(img)
    valid_x.append(csv_feature)
    valid_y.append(label)

# breakpoint()
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, TweedieRegressor
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

def fit_model(X, y, test=None, X_test=None, y_test=None):
    X, X_test = np.array(X), np.array(X_test)
    X = X.reshape(-1, 9 * 320)
    X_test = X_test.reshape(-1, 9 * 320)

    print("X shape", X.shape)

    model = Ridge(alpha=1e-3)
    model.fit(X, np.log1p(y))
    preds = np.expm1(model.predict(X_test))

    return preds, model

def fit_model_cat(X, y, test=None, X_test=None, y_test=None):
    X, X_test = np.array(X), np.array(X_test)
    X = X.reshape(-1, 9 * 320)
    X_test = X_test.reshape(-1, 9 * 320)

    print("X shape", X.shape)
    train_dataset = cb.Pool(X, y)
    model = cb.CatBoostClassifier(loss_function='MultiClass',eval_metric='TotalF1',task_type="GPU", devices='0-1')
    grid = {'learning_rate': [0.1],
            'depth': [10],
            'l2_leaf_reg': [5 ],
            'iterations': [100]}
    grid_search_result = model.grid_search(grid, train_dataset)
    print(grid_search_result)
    model.get_params()
    # model = cb.train(pool= cb.Pool(data=X,label=y), params= {'max_depth':10,'learning_rate':0.01,'n_estimators':100,'eval_metric':'TotalF1','loss_function':'MultiClass'})
    model.fit(train_dataset,
              early_stopping_rounds=50,
              plot=True,
              silent=False)
    preds = model.predict(X_test)

    return preds, model

# val_pred, model = fit_model(train_x, train_y, test=None, X_test=valid_x, y_test=valid_y)
val_pred, model = fit_model_cat(train_x, train_y, test=None, X_test=valid_x, y_test=valid_y)

def accuracy_function(real, pred):
    score = f1_score(real, pred, average='macro')
    return score

val_y = np.array(valid_y)
val_pred = np.round(val_pred)

print(val_y.shape)
print(val_pred.shape)
print(val_y)
print()
print(val_pred)
print(accuracy_function(val_y, val_pred))
print(classification_report(val_y, val_pred))