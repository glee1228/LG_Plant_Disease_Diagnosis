# -*- encoding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from datetime import datetime
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import argparse

from dataset import PlantDACON
from model import ImageModel, ImageModel2LSTMModel, ArcfaceImageModel
from loss import FocalLoss, ArcFaceLoss
from bi_tempered_loss_pytorch import bi_tempered_logistic_loss

import timm
import torch_optimizer as optim

import wandb
import logging

from torch.optim.lr_scheduler import _LRScheduler

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(cut_w // 2, W - cut_w // 2)
    cy = np.random.randint(cut_h // 2, H - cut_h // 2)

    bbx1 = cx - cut_w // 2
    bby1 = cy - cut_h // 2
    bbx2 = cx + cut_w // 2
    bby2 = cy + cut_h // 2

    return bbx1, bby1, bbx2, bby2

def cutmix(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size()[2], x.size()[3], lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def init_logger(save_dir, comment=None):
    c_date, c_time = datetime.now().strftime("%Y%m%d/%H%M%S").split('/')
    if comment is not None:
        if os.path.exists(os.path.join(save_dir, c_date, comment)):
            comment += f'_{c_time}'
    else:
        comment = c_time
    log_dir = os.path.join(save_dir, c_date, comment)
    log_txt = os.path.join(log_dir, 'log.txt')

    os.makedirs(f'{log_dir}/ckpts')
    os.makedirs(f'{log_dir}/submissions')
    os.makedirs(f'{log_dir}/comparisons')

    global logger
    logger = logging.getLogger(c_time)

    logger.setLevel(logging.INFO)
    logger = logging.getLogger(c_time)

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    h_file = logging.FileHandler(filename=log_txt, mode='a')
    h_file.setFormatter(fmt)
    h_file.setLevel(logging.INFO)
    logger.addHandler(h_file)
    logger.info(f'Log directory ... {log_txt}')
    return log_dir

def accuracy_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    acc = (pred == real).sum()/real.shape[0]
    score = f1_score(real, pred, average='macro')
    return acc, score

def train(model, train_loader, criterion, optimizer, warmup_scheduler, scheduler, scaler, epoch, wandb, args):
    model.train()
    total_train_loss = 0
    total_train_acc = 0
    total_train_score = 0
    batch_iter = tqdm(enumerate(train_loader), 'Training', total=len(train_loader), ncols=150)

    # breakpoint()
    lam = None
    label_a, label_b = None, None
    for batch_idx, batch_item in batch_iter:
        optimizer.zero_grad()
        img = batch_item['img']['image'].cuda()
        label = batch_item['label'].cuda()

        if epoch <= args.warm_epoch:
            warmup_scheduler.step()

        if args.cutmix:
            r = np.random.rand(1)
            if r < args.cutmix_prob:
                img, label_a, label_b, lam = cutmix(img, label)

            if args.amp:
                with autocast():
                    if args.environment_feature:
                        csv_feature = batch_item['csv_feature'].cuda()
                        output = model(img, csv_feature)
                    else:
                        output = model(img)
            else:
                if args.environment_feature:
                    csv_feature = batch_item['csv_feature'].cuda()
                    output = model(img, csv_feature)
                else:
                    output = model(img)
            if r < args.cutmix_prob:
                train_loss = lam * criterion(output, label_a) + (1 - lam) * criterion(output, label_b)
            else:
                train_loss = criterion(output, label)
        else:
            if args.amp:
                with autocast():
                    if args.environment_feature:
                        csv_feature = batch_item['csv_feature'].cuda()
                        output = model(img, csv_feature)
                    else:
                        output = model(img)
            else:
                if args.environment_feature:
                    csv_feature = batch_item['csv_feature'].cuda()
                    output = model(img, csv_feature)
                else:
                    output = model(img)

            if args.loss == 'btl':
                train_loss = bi_tempered_logistic_loss(output, label, t1=0.8, t2=1.4, label_smoothing=0.06)
            else:
                train_loss = criterion(output, label)

        if args.amp:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss.backward()
            optimizer.step()

        if args.scheduler == 'cycle':
            if epoch > args.warm_epoch:
                scheduler.step()

        train_acc, train_score = accuracy_function(label, output)
        total_train_loss += train_loss
        total_train_acc += train_acc
        total_train_score += train_score
        log = f'[EPOCH {epoch}] Train Loss : {train_loss.item():.4f}({total_train_loss / (batch_idx + 1):.4f}), '
        log += f'Train Acc : {train_acc.item():.4f}({total_train_acc / (batch_idx + 1):.4f}), '
        log += f'Train F1 : {train_score.item():.4f}({total_train_score / (batch_idx + 1):.4f})'
        if batch_idx+1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Train Loss : {total_train_loss / (batch_idx + 1):.4f}, '
            log += f'Train Acc : {total_train_acc / (batch_idx + 1):.4f}, '
            log += f'Train F1 : {total_train_score / (batch_idx + 1):.4f}, '
            log += f"LR : {optimizer.param_groups[0]['lr']:.2e}"


        batch_iter.set_description(log)
        batch_iter.update()



    _lr = optimizer.param_groups[0]['lr']
    train_mean_loss = total_train_loss / len(batch_iter)
    train_mean_acc = total_train_acc / len(batch_iter)
    train_mean_f1 = total_train_score / len(batch_iter)

    logger.info(log)
    batch_iter.close()

    if args.wandb:
        wandb.log({'train_mean_loss': train_mean_loss, 'lr': _lr, 'train_mean_acc': train_mean_acc, 'train_mean_f1': train_mean_f1}, step=epoch)


@torch.no_grad()
def valid(model, val_loader, criterion, epoch, wandb, args):
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    total_val_score = 0

    batch_iter = tqdm(enumerate(val_loader), 'Validating', total=len(val_loader), ncols=150)

    for batch_idx, batch_item in batch_iter:
        img = batch_item['img']['image'].cuda()
        label = batch_item['label'].cuda()

        if args.environment_feature:
            csv_feature = batch_item['csv_feature'].cuda()
            with torch.no_grad():
                output = model(img, csv_feature)
        else:
            with torch.no_grad():
                output = model(img)
        if args.loss == 'btl':
            val_loss = bi_tempered_logistic_loss(output, label, t1=0.8, t2=1.4, label_smoothing=0.06)
        else:
            val_loss = criterion(output, label)
        val_acc, val_score = accuracy_function(label, output)
        total_val_loss += val_loss
        total_val_acc += val_acc
        total_val_score += val_score

        log = f'[EPOCH {epoch}] Valid Loss : {val_loss.item():.4f}({total_val_loss / (batch_idx + 1):.4f}), '
        log += f'Valid Acc : {val_acc.item():.4f}({total_val_acc / (batch_idx + 1):.4f}), '
        log += f'Valid F1 : {val_score.item():.4f}({total_val_score / (batch_idx + 1):.4f})'
        if batch_idx+1 == len(batch_iter):
            log = f'[EPOCH {epoch}] Valid Loss : {total_val_loss / (batch_idx + 1):.4f}, '
            log += f'Valid Acc : {total_val_acc / (batch_idx + 1):.4f}, '
            log += f'Valid F1 : {total_val_score / (batch_idx + 1):.4f}'
        batch_iter.set_description(log)
        batch_iter.update()

    val_mean_loss = total_val_loss / len(batch_iter)
    val_mean_acc = total_val_acc / len(batch_iter)
    val_mean_f1 = total_val_score / len(batch_iter)
    logger.info(log)
    batch_iter.set_description(log)
    batch_iter.close()

    if args.wandb:
        wandb.log({'valid_mean_loss': val_mean_loss,'valid_mean_acc': val_mean_acc, 'valid_mean_f1': val_mean_f1}, step=epoch)

    return val_mean_loss, val_mean_acc, val_mean_f1



@torch.no_grad()
def test(model, test_loader, label_decoder, epoch, fold, wandb, args):
    model.eval()
    batch_iter = tqdm(enumerate(test_loader), 'Testing', total=len(test_loader),
                      leave=False)
    preds = []
    outputs = None
    output_path = f'{args.log_dir}/submissions/output_ep{epoch:03d}_fold{fold:02d}_{args.model}.pt'
    start = time.time()

    for i, (batch, batch_item) in enumerate(batch_iter):
        img = batch_item['img']['image'].cuda()
        if args.environment_feature:
            csv_feature = batch_item['csv_feature'].cuda()
            with torch.no_grad():
                output = model(img, csv_feature)
        else:
            with torch.no_grad():
                output = model(img)
        if i == 0:
            outputs = output.cpu()
        else:
            outputs = torch.cat((outputs, output.cpu()), 0)

        output = torch.argmax(output, dim=1).clone().cpu().numpy()
        preds.extend(output)

    preds = np.array([label_decoder[int(val)] for val in preds])
    submission = pd.read_csv(f'{args.data_path}/sample_submission.csv')
    submission['label'] = preds
    submission.to_csv(f'{args.log_dir}/submissions/submission_ep{epoch:03d}_fold{fold:02d}_{args.model}.csv', index=False)

    end = time.time()

    torch.save(outputs, output_path)
    del outputs
    if args.wandb:
        wandb.log({'preprocess-infer-save-time.': end - start}, step=epoch)

    return output_path

@torch.no_grad()
def predict_label(model, loader, label_description, label_decoder, epoch, fold, args):
    model.eval()
    batch_iter = tqdm(enumerate(loader), 'Predicting', total=len(loader),
                      leave=False)
    preds = []
    answer = []

    for batch, batch_item in batch_iter:
        img = batch_item['img']['image'].cuda()
        if args.environment_feature:
            csv_feature = batch_item['csv_feature'].cuda()
            output = model(img, csv_feature)
        else:
            output = model(img)

        output = torch.argmax(output, dim=1).clone().cpu().numpy()
        preds.extend(output)
        answer.extend(batch_item['label'])

    answer = np.array([label_description[label_decoder[int(val)]] for val in answer])
    preds = np.array([label_description[label_decoder[int(val)]] for val in preds])
    new_crosstab = pd.crosstab(answer, preds, rownames=['answer'], colnames=['preds'])
    new_crosstab.to_csv(f'{args.log_dir}/comparisons/comparison_ep{epoch:03d}_fold{fold:02d}_{args.model}.csv', index=True)
    # print(new_crosstab)



@torch.no_grad()
def ensemble_5fold_pt(model_name, output_path_list, label_decoder, args):
    predict_list = []

    for output_path in output_path_list:
        outputs = torch.load(output_path)
        preds = torch.softmax(outputs, dim=1).clone().detach().cpu().numpy()
        predict_list.append(np.array(preds))

    # ensemble = np.array(predict_list[0] + predict_list[1] + predict_list[2]) / len(predict_list)
    ensemble = np.array(predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4])/len(predict_list)

    ensemble = np.argmax(ensemble,axis=1)
    ensemble = np.array([label_decoder[val] for val in ensemble])
    submission = pd.read_csv(f'{args.data_path}/sample_submission.csv')
    submission['label'] = ensemble
    csv_name = f'submission_avg_{args.data_split.lower()}_{"_".join(model_name.split("_"))}_ep{args.epochs:03d}_'
    csv_name +=f'{args.image_size}_{args.optimizer}_lr_{args.learning_rate}_wd_{args.weight_decay}_'
    csv_name +=f'_aug_{args.aug_ver:03d}_loss_{args.loss}_amp_{args.amp}_csv_{args.environment_feature}.csv'
    submission.to_csv(csv_name, index=False)

@torch.no_grad()
def ensemble_5fold(model_name, model_path_list, test_loader, avail_features, label_encoder,label_decoder, args):
    predict_list = []
    for model_path in model_path_list:
        # Multi-modal Model
        if args.environment_feature:
            model = ImageModel2LSTMModel(model_name=args.model,
                                         pretrained_model_path=None,
                                         max_len=args.max_len,
                                         img_embedding_dim=args.img_embedding_dim,
                                         env_embedding_dim=args.env_embedding_dim,
                                         num_features=len(avail_features),
                                         class_n=len(label_encoder.keys()),
                                         dropout_rate=0,
                                         mode='test')

        # Only-image Model
        else:
            if 'arc' in args.model :
                model = ArcfaceImageModel(model_name=args.model,
                                          class_n=len(label_encoder.keys()),
                                          drop_path_rate=0,
                                          mode='test')
            else:
                model = ImageModel(model_name=args.model,
                                   class_n=len(label_encoder.keys()),
                                   drop_path_rate=0,
                                   mode='test')
        # breakpoint()
        model.load_state_dict(
            torch.load(model_path)['model_state_dict'])
        model = nn.DataParallel(model.cuda())
        model.eval()
        batch_iter = tqdm(enumerate(test_loader), f'{model_name}, {"_".join(model_path.split("/")[-3:])} Testing', total=len(test_loader), ncols=150)
        preds = []

        for batch, batch_item in batch_iter:
            img = batch_item['img']['image'].cuda()
            if args.environment_feature:
                csv_feature = batch_item['csv_feature'].cuda()
                output = model(img, csv_feature)
            else:
                output = model(img)

            pred = torch.softmax(output, dim=1).clone().detach().cpu().numpy()
            preds.extend(pred)

        predict_list.append(np.array(preds))

    ensemble = np.array(predict_list[0] + predict_list[1] + predict_list[2] + predict_list[3] + predict_list[4])/len(predict_list)

    ensemble = np.argmax(ensemble,axis=1)
    ensemble = np.array([label_decoder[val] for val in ensemble])
    submission = pd.read_csv(f'{args.data_path}/sample_submission.csv')
    submission['label'] = ensemble
    submission.to_csv(f'submission_avg_ensemble_{model_name}.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='./data',
                        help='Data root path.')
    parser.add_argument('-sp', '--save_path', type=str, default='/mldisk/nfs_shared_/dh/weights/plant-dacon',
                        help='Directory where the generated files will be stored')
    parser.add_argument('-c', '--comment', type=str, default=None)

    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='Number of epochs to train the DML network. Default: 30')
    parser.add_argument('-we', '--warm_epoch', type=int, default=0,
                        help='Number of warmup epochs to train the DML network. Default: 0')

    parser.add_argument('-bs', '--batch_size', type=int, default=14,
                        help='The size of the image you want to preprocess. '
                             'Default: 32')
    parser.add_argument('-is', '--image_size', type=int, default=384,
                        help='Variables to resize the image. '
                             'Default: 384')
    parser.add_argument('-nw', '--num_workers', type=int, default=16,
                        help='Number of workers of dataloader')

    # augmentation configs:
    parser.add_argument('-av', '--aug_ver', type=int, default=29,
                        help='Name of Data Augmentation(Refer to dataset.py). Options: "0 (No Aug) , 1 ~ 20.')
    parser.add_argument('-cm', '--cutmix', type=bool, default=False,
                        help='Cutmix Auemtnation. Default: True.')
    parser.add_argument('-cmp', '--cutmix_prob', type=float, default=0.25,
                        help='Cutmix Auemtnation Probability. Default: 0.25.')


    # loss configs:
    parser.add_argument('-l', '--loss', type=str, default='focal',
                        help='Name of loss function. Options: "ce, focal, btl, arcface')

    parser.add_argument('-cw', '--class_weights', type=bool, default=False)

    # optimizer configs:
    parser.add_argument('-ot', '--optimizer', type=str, default='adamw',
                        help='Name of Optimizer. Options: "adam, radam, adamw, adamp, ranger, lamb')
    parser.add_argument('-sc', '--scheduler', type=str, default='cos_base',
                        help='Name of Optimizer. Options: "cos_base, cos, cos_warm_restart, cycle')
    # Adam
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate of the DML network. Default: 10^-4')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3,
                        help='Regularization parameter of the DML network. Default: 10^-4')

    # Step
    parser.add_argument('-st', '--step_size', type=int, default=3)
    parser.add_argument('-sg', '--step_gamma', type=float, default=0.8)

    # Cycle
    parser.add_argument('-mal', '--max_lr', type=float, default=1e-3,
                        help='Regularization parameter of the DML network. Default: 10^-4')

    # Cosine Annealing
    parser.add_argument('-tm', '--tmax', type=int, default=60,
                        help='Regularization parameter of the DML network. Default: 10^-4')
    parser.add_argument('-mil', '--min_lr', type=float, default=1e-6,
                        help='Regularization parameter of the DML network. Default: 10^-4')

    # data split configs:
    parser.add_argument('-ds', '--data_split', type=str, default='StratifiedKFold',
                        help='Name of Training Data Sampling Strategy. Options: "Split_base, Stratified, StratifiedKFold, KFold')
    parser.add_argument('-ns', '--n_splits', type=int, default=5,
                        help='The number of datasets(Train,val) to be divided.')
    parser.add_argument('-rs', '--random_seed', type=int, default=42,
                        help='Random Seed')
    parser.add_argument('-vr', '--val_ratio', type=float, default=0.2,
                        help='validation dataset ratio')


    # image model specific configs:
    parser.add_argument('-m', '--model', type=str, default='arc_convnext_xlarge_384_in22ft1k',
                        help='Name of model. Options: Refer to image_model_list.txt. '
                             'Option : "convnext_large_384_in22ft1k, swin_large_patch4_window12_384_in22k'
                             'arc_convnext_large_384_in22ft1k, arc_swin_large_patch4_window12_384_in22k'
                             'If you want to use arcface loss while learning only the image model, '
                             'set the "arc_xxx" model and the "arcface" loss function and environment feature to "FALSE".')

    parser.add_argument('-dpr', '--drop_path_rate', type=float, default=0.2,
                        help='dropout rate')

    # multi-modal model specific configs:
    parser.add_argument('-ied', '--img_embedding_dim', type=int, default=1024,
                        help='Dimension of the output embedding of image model')
    parser.add_argument('-eed', '--env_embedding_dim', type=int, default=1024,
                        help='Dimension of the of environment feature model')

    parser.add_argument('-ml', '--max_len', type=int, default=320,
                        help='Number of layers')
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0.2,
                        help='dropout rate')

    # feature configs:
    parser.add_argument('-ef', '--environment_feature', type=bool, default=True,
                        help='Whether to use environmental features'
                        )

    # eval configs:
    parser.add_argument('-d', '--dataset', type=str, default='PlantDACON',
                        help='Name of evaluation dataset. Options: "PlantDACON, PlantVillage')

    # wandb config:
    parser.add_argument('--wandb', type=bool, default=True,
                        help='wandb'
                        )

    # ensemble config:
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble mode'
                        )

    # amp config:
    parser.add_argument('--amp', type=bool, default=True,
                        help='amp mode'
                        )

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()


    # Data Path
    train_data = sorted(glob(f'{args.data_path}/train/*'))
    csv_files = sorted(glob(f'{args.data_path}/train/*/*.csv'))
    test_data = sorted(glob(f'{args.data_path}/test/*'))
    labelsss = pd.read_csv(f'{args.data_path}/train.csv')['label']

    folds = []
    # Data Split
    if args.data_split.lower() == 'split_base':
        train_data, val_data = train_test_split(train_data, random_state=args.random_seed, test_size=args.val_ratio, shuffle=True)
        folds.append((train_data, val_data))
        args.n_split = 1
    elif args.data_split.lower() == 'stratified':
        train_data, val_data = train_test_split(train_data, random_state=args.random_seed, test_size=args.val_ratio, stratify=labelsss, shuffle=True)
        folds.append((train_data, val_data))
        args.n_split = 1
    elif args.data_split.lower() == 'stratifiedkfold':
        train_data = np.array(train_data)
        skf = StratifiedKFold(n_splits=args.n_splits, random_state=args.random_seed, shuffle=True)
        for train_idx, valid_idx in skf.split(train_data,labelsss.tolist()):
            folds.append((train_data[train_idx].tolist(), train_data[valid_idx].tolist()))

    elif args.data_split == 'kfold':
        train_data = np.array(train_data)
        kf = KFold(n_splits=args.n_splits, random_state=args.random_seed, shuffle=True)
        for train_idx, valid_idx in kf.split(train_data,labelsss.tolist()):
            folds.append((train_data[train_idx].tolist(), train_data[valid_idx].tolist()))
    else:
        pass

    # Available CSV feature for Training and Testing
    if args.environment_feature:
        avail_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고',
                        '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']
    else:
        avail_features = None

    # Label Description : Refer to CSV File
    label_description = {
        "1_00_0": "딸기",
        "2_00_0": "토마토",
        "2_a5_2": "토마토_흰가루병_중기",
        "3_00_0": "파프리카",
        "3_a9_1": "파프리카_흰가루병_초기",
        "3_a9_2": "파프리카_흰가루병_중기",
        "3_a9_3": "파프리카_흰가루병_말기",
        "3_b3_1": "파프리카_칼슘결핍_초기",
        "3_b6_1": "파프리카_다량원소결필(N)_초기",
        "3_b7_1": "파프리카_다량원소결필(P)_초기",
        "3_b8_1": "파프리카_다량원소결필(K)_초기",
        "4_00_0": "오이",
        "5_00_0": "고추",
        "5_a7_2": "고추_탄저병_중기",
        "5_b6_1": "고추_다량원소결필(N)_초기",
        "5_b7_1": "고추_다량원소결필(P)_초기",
        "5_b8_1": "고추_다량원소결필(K)_초기",
        "6_00_0": "시설포도",
        "6_a11_1": "시설포도_탄저병_초기",
        "6_a11_2": "시설포도_탄저병_중기",
        "6_a12_1": "시설포도_노균병_초기",
        "6_a12_2": "시설포도_노균병_중기",
        "6_b4_1": "시설포도_일소피해_초기",
        "6_b4_3": "시설포도_일소피해_말기",
        "6_b5_1": "시설포도_축과병_초기"
    }


    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}

    # Ensemble Inference & Save Result
    if args.ensemble :
        test_dataset = globals()[args.dataset](args.image_size, test_data, csv_files, avail_features, label_description, mode='test')

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*8,
                                                  num_workers=args.num_workers, shuffle=False)
        model_path_list = ['/mldisk/nfs_shared_/my/plant-dacon/weights/20220120/randomcrop/ckpts/ckpt_epoch_030_fold_0.pt',
                           '/mldisk/nfs_shared_/my/plant-dacon/weights/20220120/randomcrop_142306/ckpts/ckpt_epoch_030_fold_1.pt',
                           '/mldisk/nfs_shared_/my/plant-dacon/weights/20220120/randomcrop_152402/ckpts/ckpt_epoch_019_fold_2.pt',
                           '/mldisk/nfs_shared_/my/plant-dacon/weights/20220120/randomcrop_162459/ckpts/ckpt_epoch_026_fold_3.pt',
                           '/mldisk/nfs_shared_/my/plant-dacon/weights/20220120/randomcrop_172601/ckpts/ckpt_epoch_020_fold_4.pt']
        ensemble_5fold(args.model, model_path_list, test_loader, avail_features, label_encoder, label_decoder, args)
        exit()

    if args.class_weights :
        val_counts = labelsss.value_counts().sort_index().values
        class_weights = 1/np.log1p(val_counts)
        class_weights = (class_weights / class_weights.sum()) * len(label_encoder.keys())
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None


    # Criterion
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'btl':
        criterion = None
    elif args.loss == 'arcface':
        criterion = ArcFaceLoss(criterion=FocalLoss(), weight=class_weights)
        # criterion = ArcFaceLoss(criterion=FocalLoss())

    print(f'The number of datasets separated : {len(folds)}')
    best_model_paths = []
    output_path_list = []
    best_image_model_paths = [None]*len(folds)

    ### convnext-large best model(arcface-convnext-focal)
    # best_image_model_paths = ['/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/153332/ckpts/ckpt_epoch_014_fold_0.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/164824/ckpts/ckpt_epoch_014_fold_1.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/180333/ckpts/ckpt_epoch_027_fold_2.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/191922/ckpts/ckpt_epoch_025_fold_3.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/203426/ckpts/ckpt_epoch_023_fold_4.pt']

    ### Swin-base best model(arcface-swin-focal-aug_24)
    # best_image_model_paths = ['/mldisk/nfs_shared_/dh/weights/plant-dacon/20220129/200755/ckpts/ckpt_epoch_027_fold_0.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220129/210914/ckpts/ckpt_epoch_019_fold_1.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220129/221049/ckpts/ckpt_epoch_025_fold_2.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220129/231213/ckpts/ckpt_epoch_020_fold_3.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/001346/ckpts/ckpt_epoch_025_fold_4.pt']

    ### Swin-base best model(arcface-swin-focal-aug_29)
    # best_image_model_paths = ['/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/221808/ckpts/ckpt_epoch_028_fold_0.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220130/232028/ckpts/ckpt_epoch_026_fold_1.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220131/002403/ckpts/ckpt_epoch_029_fold_2.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220131/012727/ckpts/ckpt_epoch_023_fold_3.pt',
    #                          '/mldisk/nfs_shared_/dh/weights/plant-dacon/20220131/023022/ckpts/ckpt_epoch_028_fold_4.pt']

    for fold in range(len(folds)):
        train_data, val_data = folds[fold]

        # Multi-modal Model
        if args.environment_feature:
            model = ImageModel2LSTMModel(model_name=args.model,
                                         pretrained_model_path=best_image_model_paths[fold],
                                         max_len=args.max_len,
                                         img_embedding_dim=args.img_embedding_dim,
                                         env_embedding_dim=args.env_embedding_dim,
                                         num_features=len(avail_features),
                                         class_n=len(label_encoder.keys()),
                                         dropout_rate=0.1,
                                         mode='train')

        # Only-image Model
        else:
            if 'arc' in args.model:
                model = ArcfaceImageModel(model_name=args.model,
                                          class_n=len(label_encoder.keys()),
                                          drop_path_rate=args.drop_path_rate,
                                          mode='train')
            else:
                model = ImageModel(model_name=args.model,
                                   class_n=len(label_encoder.keys()),
                                   drop_path_rate=args.drop_path_rate,
                                   mode='train')

        model = nn.DataParallel(model.cuda())

        # Dataset
        train_dataset = globals()[args.dataset](args.image_size, train_data, csv_files, avail_features, label_description, args.aug_ver, mode='train')
        val_dataset = globals()[args.dataset](args.image_size, val_data, csv_files, avail_features, label_description, mode='valid')



        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                 num_workers=args.num_workers, shuffle=False)


        # Optimizer & Scheduler Setting
        optimizer = None
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        elif args.optimizer == 'radam':
            optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        elif args.optimizer == 'ranger':
            optimizer = optim.Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)
        elif args.optimizer == 'lamb':
            optimizer = optim.Lamb(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        else:
            pass

        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm_epoch)
        scheduler = None
        if args.scheduler == 'cos_base':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == 'cos_warm_restart':
            args.epochs = 69
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, last_epoch=-1)
        elif args.scheduler == 'cos':
             # tmax = epoch * 2 => half-cycle
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.min_lr)
        elif args.scheduler == 'cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=iter_per_epoch, epochs=args.epochs)


        # amp scaler
        scaler = None
        if args.amp:
            scaler = GradScaler()

        # log dir setting
        log_dir = init_logger(args.save_path, args.comment)
        args.log_dir = log_dir

        # Wandb initialization
        run = None
        if args.wandb:
            c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')
            run = wandb.init(project=args.dataset, name=f'{args.model}_{c_date}_{c_time}_fold_{fold}')
            wandb.config.update(args)

        best_vacc, best_f1 = .0, .0
        best_vloss = 9999.
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, criterion, optimizer, warmup_scheduler, scheduler, scaler, epoch, wandb, args)
            vloss, vacc, vf1 = valid(model, val_loader, criterion, epoch, wandb, args)
            predict_label(model, val_loader, label_description, label_decoder, epoch, fold, args)
            if vf1 > best_f1:
                best_epoch = epoch
                best_vloss = min(vloss, best_vloss)
                best_f1 = max(vf1, best_f1)
                if best_f1 > 0.9:
                    torch.save({'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'epoch': epoch, },
                               f'{log_dir}/ckpts/ckpt_epoch_{epoch:03d}_fold_{fold:01d}.pt')

            if args.scheduler in ['cos_base', 'cos', 'cos_warm_restart']:
                if epoch > args.warm_epoch:
                    scheduler.step()

        best_model_paths.append(f'{log_dir}/ckpts/ckpt_epoch_{best_epoch:03d}_fold_{fold:01d}.pt')

        del model
        del optimizer, scheduler
        del train_dataset, val_dataset

        ### Best Model Inference
        # Multi-modal Model
        if args.environment_feature:
            model = ImageModel2LSTMModel(model_name=args.model,
                                         pretrained_model_path=best_image_model_paths[fold],
                                         max_len=args.max_len,
                                         img_embedding_dim=args.img_embedding_dim,
                                         env_embedding_dim=args.env_embedding_dim,
                                         num_features=len(avail_features),
                                         class_n=len(label_encoder.keys()),
                                         dropout_rate=0,
                                         mode='test')

        # Only-image Model
        else:
            if 'arc' in args.model :
                model = ArcfaceImageModel(model_name=args.model,
                                          class_n=len(label_encoder.keys()),
                                          drop_path_rate=0,
                                          mode='test')
            else:
                model = ImageModel(model_name=args.model,
                                   class_n=len(label_encoder.keys()),
                                   drop_path_rate=0,
                                   mode='test')
        model.load_state_dict(torch.load(f'{log_dir}/ckpts/ckpt_epoch_{best_epoch:03d}_fold_{fold:01d}.pt')['model_state_dict'])
        model = nn.DataParallel(model.cuda())

        test_dataset = globals()[args.dataset](args.image_size, test_data, csv_files, avail_features,
                                               label_description, mode='test')
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, shuffle=False)
        output_path = test(model, test_loader, label_decoder, best_epoch, fold, wandb, args)
        output_path_list.append(output_path)
        del test_dataset
        del model

        if args.wandb:
            run.finish()

    if args.data_split.lower() in ['stratifiedkfold','kfold'] :
        ensemble_5fold_pt(model_name=args.model, output_path_list=output_path_list, label_decoder=label_decoder, args=args)

if __name__ == '__main__':
    main()









