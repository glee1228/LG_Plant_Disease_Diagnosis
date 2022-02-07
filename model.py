import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import timm
import torch.nn.functional as F


class ImageModel(nn.Module):
    def __init__(self, model_name, class_n, drop_path_rate, mode='train'):
        super().__init__()
        self.model_name = model_name.lower()
        self.class_n = class_n
        self.drop_path_rate = drop_path_rate
        self.mode = mode
        # 모델
        if self.model_name == 'resnet50':
            self.encoder = Resnet50(class_n=class_n)
        elif self.model_name == 'convnext_large_384_in22ft1k':
            if self.mode == 'train' :
                self.encoder = timm.models.convnext_large_384_in22ft1k(pretrained=True, drop_path_rate=self.drop_path_rate)
            else:
                self.encoder = timm.models.convnext_large_384_in22ft1k(pretrained=True)
        elif self.model_name == 'convnext_base_384_in22ft1k':
            if self.mode == 'train' :
                self.encoder = timm.models.convnext_base_384_in22ft1k(pretrained=True, drop_path_rate=self.drop_path_rate)
            else:
                self.encoder = timm.models.convnext_base_384_in22ft1k(pretrained=True)
        else:
            if self.drop_path_rate != 0 :
                if self.mode == 'train' :
                    self.encoder = timm.create_model(self.model_name, pretrained=True, drop_path_rate=self.drop_path_rate)
                else:
                    self.encoder = timm.create_model(model_name, pretrained=True)
            else:
                self.encoder = timm.create_model(model_name, pretrained=True)

        names = []
        modules = []
        for name, module in self.encoder.named_modules():
            names.append(name)
            modules.append(module)

        self.fc_in_features = self.encoder.num_features
        print(f'The layer was modified...')

        fc_name = names[-1].split('.')

        if len(fc_name)==1:
            print(f'{getattr(self.encoder,fc_name[0])} -> Linear(in_features={self.fc_in_features}, out_features={class_n}, bias=True)')
            setattr(self.encoder, fc_name[0], nn.Linear(self.fc_in_features, class_n))
        elif len(fc_name)==2:
            print(f'{getattr(getattr(self.encoder,fc_name[0]),fc_name[1])} -> Linear(in_features={self.fc_in_features}, out_features={class_n}, bias=True)')
            setattr(getattr(self.encoder,fc_name[0]), fc_name[1], nn.Linear(self.fc_in_features, class_n))

    def forward(self, x):
        x = self.encoder(x)

        return x

class LSTM_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, cnn_features_len, class_n, rate):
        super(LSTM_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        # self.lstm_fc = nn.Linear(embedding_dim * 2 if bidirectional else embedding_dim, 2048)
        self.lstm_fc = nn.Linear(num_features*embedding_dim, 2048)
        self.final_layer = nn.Linear(cnn_features_len + 2048, class_n)
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        self.lstm.flatten_parameters()
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.lstm_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1) # enc_out + hidden
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output


class Resnet50(nn.Module):
    def __init__(self, class_n):
        super(Resnet50, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, class_n)

    def forward(self, x):
        x = self.base(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcfaceImageModel(nn.Module):
    def __init__(self, model_name, class_n, drop_path_rate, embedding_dim=1024, mode='train', encode=False):
        super().__init__()
        self.model_name = '_'.join(model_name.lower().split('_')[1:])
        self.class_n = class_n
        self.drop_path_rate = drop_path_rate
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.encode = encode
        # 모델
        if self.model_name == 'resnet50':
            self.encoder = Resnet50(class_n=class_n)
        elif self.model_name == 'convnext_xlarge_384_in22ft1k':
            if self.mode == 'train' :
                self.encoder = timm.models.convnext_xlarge_384_in22ft1k(pretrained=True, drop_path_rate=self.drop_path_rate)
            else:
                self.encoder = timm.models.convnext_xlarge_384_in22ft1k(pretrained=True)
        elif self.model_name == 'convnext_large_384_in22ft1k':
            if self.mode == 'train' :
                self.encoder = timm.models.convnext_large_384_in22ft1k(pretrained=True, drop_path_rate=self.drop_path_rate)
            else:
                self.encoder = timm.models.convnext_large_384_in22ft1k(pretrained=True)
        elif self.model_name == 'convnext_base_384_in22ft1k':
            if self.mode == 'train' :
                self.encoder = timm.models.convnext_base_384_in22ft1k(pretrained=True, drop_path_rate=self.drop_path_rate)
            else:
                self.encoder = timm.models.convnext_base_384_in22ft1k(pretrained=True)
        else:
            if self.drop_path_rate != 0 :
                if self.mode == 'train' :
                    self.encoder = timm.create_model(self.model_name, pretrained=True, drop_path_rate=self.drop_path_rate)
                else:
                    self.encoder = timm.create_model(self.model_name, pretrained=True)
            else:
                self.encoder = timm.create_model(self.model_name, pretrained=True)

        names = []
        modules = []
        for name, module in self.encoder.named_modules():
            names.append(name)
            modules.append(module)

        self.fc_in_features = self.encoder.num_features
        print(f'The layer was modified...')

        fc_name = names[-1].split('.')

        if len(fc_name)==1:
            print(f'{getattr(self.encoder,fc_name[0])} -> Linear(in_features={self.fc_in_features}, out_features={1000}, bias=True)')
            setattr(self.encoder, fc_name[0], nn.Linear(self.fc_in_features, 1000))
        elif len(fc_name)==2:
            print(f'{getattr(getattr(self.encoder,fc_name[0]),fc_name[1])} -> Linear(in_features={self.fc_in_features}, out_features={1000}, bias=True)')
            setattr(getattr(self.encoder,fc_name[0]), fc_name[1], nn.Linear(self.fc_in_features, 1000))

        self.neck = nn.Sequential(
            nn.Linear(1000, self.embedding_dim, bias=True),
            nn.BatchNorm1d(self.embedding_dim),
            torch.nn.ReLU()
        )

        self.arc_margin_product = ArcMarginProduct(self.embedding_dim, self.class_n)


    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        logits = self.arc_margin_product(x)
        if self.encode :
            return x
        else:
            return logits

class ImageModel2LSTMModel(nn.Module):
    def __init__(
            self,
            model_name,
            pretrained_model_path,
            max_len,
            img_embedding_dim,
            env_embedding_dim,
            num_features,
            class_n,
            dropout_rate=0.1,
            mode='train'
    ):
        super(ImageModel2LSTMModel, self).__init__()
        self.model_name = model_name
        self.pretrained_model_path = pretrained_model_path
        self.mode = mode
        self.class_n = class_n
        self.dropout_rate = dropout_rate
        self.max_len = max_len
        self.img_embedding_dim = img_embedding_dim
        self.env_embedding_dim = env_embedding_dim
        self.num_features = num_features

        # When using new data (existing data + aihub pepper white powder data), 25 when it's not 28
        if self.pretrained_model_path :
            self.encoder = ArcfaceImageModel(model_name, 25, drop_path_rate=0, embedding_dim=self.img_embedding_dim,
                                             mode='test', encode=True)
            self.encoder.load_state_dict(torch.load(self.pretrained_model_path)['model_state_dict'])
            self.encoder.requires_grad = False
        else:
            self.encoder = ArcfaceImageModel(model_name, 25, drop_path_rate=0.2, embedding_dim=self.img_embedding_dim,
                                             mode='train', encode=True)

        self.rnn = LSTM_Decoder(self.max_len, self.env_embedding_dim, self.num_features, cnn_features_len=self.img_embedding_dim, class_n=self.class_n, rate=self.dropout_rate)

    def forward(self, img, seq):
        cnn_output = self.encoder(img)
        output = self.rnn(cnn_output, seq)
        return output



