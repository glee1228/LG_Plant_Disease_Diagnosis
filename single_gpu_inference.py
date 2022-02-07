# -*- encoding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from glob import glob
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score
import argparse

from dataset import PlantDACON
from model import ImageModel, ImageModel2LSTMModel, ArcfaceImageModel

import logging



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
    csv_name = f'submission_avg_{args.data_split.lower()}_{"_".join(model_name.split("_"))}_ep{args.epochs:03d}_'
    csv_name +=f'{args.image_size}_{args.optimizer}'
    csv_name +=f'_aug_{args.aug_ver:03d}_loss_{args.loss}_amp_{args.amp}_csv_{args.environment_feature}.csv'
    submission.to_csv(csv_name, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data_path', type=str, default='./data',
                        help='Data root path.')
    parser.add_argument('-sp', '--save_path', type=str, default='./weight/plant-lg-dacon',
                        help='Directory where the generated files will be stored')
    parser.add_argument('-c', '--comment', type=str, default=None)

    parser.add_argument('-bs', '--batch_size', type=int, default=28,
                        help='The size of the image you want to preprocess. '
                             'Default: 28')
    parser.add_argument('-is', '--image_size', type=int, default=384,
                        help='Variables to resize the image. '
                             'Default: 384')
    parser.add_argument('-nw', '--num_workers', type=int, default=16,
                        help='Number of workers of dataloader')


    # image model specific configs:
    parser.add_argument('-m', '--model', type=str, default='arc_convnext_xlarge_384_in22ft1k',
                        help='Name of model. Options: Refer to image_model_list.txt. '
                             'Option : "convnext_large_384_in22ft1k, swin_large_patch4_window12_384_in22k'
                             'arc_convnext_large_384_in22ft1k, arc_swin_large_patch4_window12_384_in22k'
                             'If you want to use arcface loss while learning only the image model, '
                             'set the "arc_xxx" model and the "arcface" loss function and environment feature to "FALSE".'
                             'If you want to use focal loss while learning the multi-modal model, ' 
                             'set the "arc_xxx" model and the "focal" loss function and environment feature to "True".')

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
    csv_files = sorted(glob(f'{args.data_path}/train/*/*.csv'))
    test_data = sorted(glob(f'{args.data_path}/test/*'))



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
    test_dataset = globals()[args.dataset](args.image_size, test_data, csv_files, avail_features, label_description, mode='test')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*8,
                                              num_workers=args.num_workers, shuffle=False)

    model_path_list = ['/workspace/weights/plant-dacon/20220204/201544/ckpts/ckpt_epoch_024_fold_0.pt',
                       '/workspace/weights/plant-dacon/20220204/225438/ckpts/ckpt_epoch_029_fold_1.pt',
                       '/workspace/weights/plant-dacon/20220205/013444/ckpts/ckpt_epoch_017_fold_2.pt',
                       '/workspace/weights/plant-dacon/20220205/041456/ckpts/ckpt_epoch_020_fold_3.pt',
                       '/workspace/weights/plant-dacon/20220205/065515/ckpts/ckpt_epoch_023_fold_4.pt']

    ensemble_5fold(args.model, model_path_list, test_loader, avail_features, label_encoder, label_decoder, args)


if __name__ == '__main__':
    main()









