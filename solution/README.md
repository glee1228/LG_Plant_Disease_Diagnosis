## 솔루션 

전반적인 전략은 고도로 조정된 이미지 모델을 취하는 것이었습니다.

우선 대부분의 다른 참가자분들과 마찬가지로 1) 이미지 모델을 학습하는데 많은 제출을 사용 2) 새로운 접근 방식 모색 3) 절제 연구를 지향했습니다.

* 이미지만 사용한 resnet50의 경우 seed를 고정한 동일한 fold의 validation 검증 과정에서 f1-macro 기준 0.91, 환경 변수만 사용한 CatBoostClassifier의 경우 0.86였고 이미지 모델이 파프리카 흰가루병의 진행 정도(초기, 중기, 말기)에서 더 변별력 있는 성능을 보여주어 이미지 모델을 Ablation Study를 진행할 주 모델로 선정했습니다.(CatBoostClassifier는 파프리카 흰가루병의 3개 클래스에 한해서 f1-macro 0.33으로 분류)

* PlantVillage 벤치마크에서는 병해 분류에 Lightly Augmentation+Small Image Model을 활용한 연구가 주로 진행되고 있음을 확인했지만, 제 실험의 경우 Heavily Augmentation+Large Image Model이 상당한 성능 향상을 가져왔습니다.

* Kaggle을 참고하여 작물 병해 관련 데이터에서 높은 성능을 보이는 Augmentation 요소 후보를 선별하고 조합을 구성했습니다. Augmentation의 조합을 Wandb Tool을 이용해 이미지 모델의 Cross Validation 성능으로 비교적 좋은 Augmentation 조합을 제출하여 public LB 점수를 평가하였습니다.

동시에 ImageNet 벤치마크에서 크기와 성능의 trade-off를 고려하여 이미지 모델을 선별하여 훈련을 시도했습니다.(regnety_040, efficientnet_b7_ns, efficientnetv2, swin_base_transformer,swin_large_transformer, beit_large, convnext_large, convnext_xlarge)

6천장의 작물의 Train 데이터를 학습해 한 번도 본 적 없는 훨씬 많은 양의 작물의 Test 데이터를 분류해야하는 Task가 Open-set Face Recognition Task와 비슷하다고 생각해 Face Recognition과 Landmark Recognition에 많이 사용되는 Angular Margin을 이용한 학습을 시도했습니다.

캐글의 Cassava Classification 1등 솔루션에서 채택한 Swin-transformer+Bi-tempered-Loss보다 Swin-Transformer+Arcface-Loss가 효과적으로 작물 병해 이미지를 Discriminative한 feature space로 매핑하는 것을 public LB를 통해 잠정적으로 알 수 있었습니다.

그래서 이미지 모델을 Arcface Loss를 이용해 Pretrain한 뒤 LSTM만 Fine-tuning하는 방법을 시도했습니다.

결론적으로는 public LB 점수가 더 높았던 End-to-end로 학습한 이미지 모델+LSTM 구조를 채택했습니다. 

#### 최종 제출물은 ConvNext+LSTM 구조의 5-fold 누적 평균을 이용한 Ensemble 입니다.
![figure](https://github.com/glee1228/LG_Plant_Disease_Diagnosis/blob/main/solution/figure.png) 
public LB의 점수는 0.95165, private LB의 점수는 0.95377 입니다.

나머지 구조는 Public LB를 통해 평가되었습니다.

저는 "convnext_xlarge_384_in22ft1k+Vanilla LSTM" 구조의 ```ConvNext과 LSTM 모델```을 다음 구성으로 사용했습니다.

* ImageNet-22K에 pretrain된 가중치 로드
* (384, 384)의 이미지 크기
* Train Augmentation : Resize, ShiftScaleRotate, HorizontalFlip, VerticalFlip, RandomRotate90, CLAHE, Sharpen, RandomBrightnessContrast, RandomResizedCrop, Normalize
* Valid Augmentation : Resize, HorizontalFlip, VerticalFlip, Normalize
* Test Augmentation : Resize, Normalize
* 320 길이의 환경 변수 데이터(온도, 습도, 이슬점의 평균, 최고, 최저)
* Embedding Vector의 크기는 LSTM : 2048, Image Model : 1024
* gamma=2.0인 Focal Loss
* 30 epoch까지 2e-7까지 learning rate가 떨어지는 CosineAnnealingLR 스케줄러를 사용한 Adamw 옵티마이저(lr=1e-4,weigt decay=1e-3)
* 30 epoch의 5-fold-CV(validation F1-macro Score가 가장 높은 모델을 선택)
* AMP를 이용한 학습


단일 모델의 5개 fold의 예측 확률을 누적 평균한 예측 결과를 최종 결과로 제출했습니다.

## Abalation Study

Arcitecture	| Ensemble	| Augmentation |	Loss	 | public LB |	private LB | note |
---- |  ---- | ---- | ---- | ----  |---- |----
regnety_040  |	Single  |	None	 | CE  | 	0.91144 | 	-
regnety_040  |	5-fold  |	None	 | CE  | 	0.92673 | 	-  | + 0.015
regnety_040  |	5-fold  |	Flip	 | CE  | 	0.91809 | 	-
regnety_040  |	5-fold  |	Flip & Contrast	 | CE  | 	0.92431 | 	-
regnety_040  |	5-fold  |	Flip & SSR	 | CE  | 	0.91624 | 	-
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR	 | bi-tempered  | 	0.94607 | 	-  | + 0.03
swin_base_patch4_window12_384_in22k  |	5-fold |	Flip & SSR & Transpose	 | bi-tempered  | 	0.94169 | 	-
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR & HSV | bi-tempered  | 	0.94080 | 	-
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR & RandomBrightnessContrast | bi-tempered  | 	0.94592 | 	-
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR & CLAHE | bi-tempered  | 	0.94641 | 	-
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR & CLAHE & Sharpen | bi-tempered  | 	0.94389 | 	-
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC | bi-tempered  | 	0.94771 | -  | + 0.02
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC | CE  | 	0.9487 | -  | + 0.01
swin_base_patch4_window12_384_in22k  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC | Arcface  | 	0.94952 | 	-  | + 0.01
swin_base_patch4_window12_384_in22k + LSTM(fine-tuning)  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC | Arcface+Focal  | 	0.95093 | 	-  | + 0.01
swin_large_patch4_window12_384_in22k + LSTM(fine-tuning)  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC & RandomRotate90 | Arcface+Focal  | 	0.94372 | 	-
swin_large_patch4_window12_384_in22k + LSTM(End-to-end)  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC & RandomRotate90  | Focal  | 	0.95276 | 	0.95723 | + 0.02  (Best)
convnext_large_384_in22ft1k  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC | bi-tempered  | 	0.94918 | 	-
convnext_large_384_in22ft1k  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC | CE  | 	0.94949 | 	-
convnext_large_384_in22ft1k  +LSTM(End-to-end)  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC & RandomRotate90| Focal  | 	0.94996 | 	-
convnext_xlarge_384_in22ft1k  +LSTM(End-to-end)  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC & RandomRotate90 | Focal  | 	0.95165 | 	0.95377 | Submission
convnext_xlarge_384_in22ft1k  +LSTM(End-to-end)  |	5-fold  |	Flip & SSR & CLAHE & Sharpen & RBC & RRC & RandomRotate90 | Focal  | 	0.95431 | 	0.95553 | Unofficial

* SSR : ShiftScaleRotate, RBC : RandomBrightnessContrast, RRC : RandomResizedCrop

**이슈 -**
최종 제출 예정이었던 구조(swin_large_transformer+LSTM)의 public LB의 점수가 0.95276(private LB : 0.95723)으로 가장 높았지만 도커 컨테이너의 가중치 파일을 로컬 서버로 deployment하는 과정에서 학습한 가중치 파일이 손상되어 그 다음 public LB 점수가 높았던 모델(Convnext_xlarge+LSTM)의 public LB를 최종 제출했습니다.

***시도했지만 성능 개선이 이루어지지 않아 채택하지 않은 것들***
* Augmentation : MixUp, CutMix, Transpose, GridDistortion, HueSaturationValue
* Loss : Bi-tempered Logistic Loss, Arcface Loss(이미지 모델 pretrain)+Focal Loss(시계열 모델 fine-tuning)
* Scheduler : CosineAnnealingWarmRestarts(69에폭 동안 3순회)
* Architecture : regnety_040, efficientnet_b7_ns, efficientnetv2, swin_base_transformer, beit_large, convnext_xlarge
* Data :
1. 파프리카 초기, 중기 말기 클래스 특징을 육안으로 파악하여 3개의 클래스를 동일한 비율로 재 분배 
2. Ai-hub의 노지작물 데이터 중 고추(Pepper) 흰가루병 이미지도 초기, 중기, 말기로 구성되어 있고 파프리카 흰가루병과 특징 상 비슷하여 3개 클래스(고추 흰가루병 초기,중기,말기)를 추가하여 28개 클래스로 이미지 모델 학습 - 미미한 효과









