# 농업 환경 변화에 따른 작물 병해 진단 AI 경진대회

### Private score 4th 0.953772

* 주최 : LG AI Research
* 주관 : DACON
* https://dacon.io/competitions/official/235870/overview/description



### Development Environment
Ubuntu 18.04.5 LTS

### Library Version
* h5py>=2.10.0
* numpy>=1.18.1
* tqdm>=4.43.0
* albumentations==1.1.0
* matplotlib==3.5.1
* opencv-python-headless==4.5.5.62
* pandas==1.3.5
* Pillow==9.0.0
* scikit-image==0.19.1
* scikit-learn==1.0.2
* scipy==1.7.3
* timm==0.5.4
* torch==1.8.0
* torch-optimizer==0.3.0
* torchvision==0.9.0
* wandb==0.12.9



### Directory Structure
```
/workspace
├── data
│   ├── train
│   │    ├── 10027
│   │         ├── 10027.csv
│   │         ├── 10027.jpg
│   │         └── 10027.json
│   │    ├── ...
│   │    └── 67678
│   ├── test
│   │    ├── 10000
│   │    ├── ...
│   │    └── 67677
│   │    
│   ├── train.csv
│   └── sample_submission.csv
│
├── baseline.py (실행 코드)
├── dataset.py (데이터셋 클래스)
├── model.py (모델 클래스)
├── image_model_list.txt (참고 : 사용 가능한 이미지 모델 이름)
├── requirement.txt
├── Dockerfile   
└── docker-compose.yml
```

### (Recommended) Docker-compose Usage
1. `git clone https://github.com/glee1228/LG_Plant_Disease_Diagnosis.git`

2. Edit `docker-compose.yml`
    ```
    services:
      main:
        container_name: plant-lg-dacon
        ...
        ports:
          - "{host ssh}:22"
        ipc: host
        stdin_open: true
    ```
3. Download data.zip from https://dacon.io/competitions/official/235870/data to container workspace data path.
    ```bash
    #./LG_Plant_Disease_Diagnosis
    mkdir data
    cd data
    (Download data to ./LG_Plant_Disease_Diagnosis/data/)
    ```

4. Build docker image clearly and create containers
    ```bash
    #./LG_Plant_Disease_Diagnosis
    docker-compose build --no-cache
    docker-compose up -d
    docker attach plant-lg-dacon
    ```
    
5. Unzip train, test data
    ```bash
    #/workspace/data
    unzip data.zip
    unzip train.zip
    unzip test.zip
    ```
    
6. (Option) Set password and Restart SSH for SFTP connection
    ```bash
    passwd
    /etc/init.d/ssh restart
    ```
    
7. Train  `baseline.py`
    ```bash
    #/workspace
    python baseline.py
    ```

8. Submit 
`/workspace/submission_xxx.csv`



### (Option) Jupyter Notebook Usage
1. Install Library
    ```
    pip3 install -r requirement.txt
    pip3 install jupyter
    ```


2. Download data.zip from https://dacon.io/competitions/official/235870/data to container workspace data path.
    ```bash
    #./LG_Plant_Disease_Diagnosis
    mkdir data
    cd data
    (Download data to ./LG_Plant_Disease_Diagnosis/data/)
    ```
3. Unzip train, test data
    ```bash
    #./LG_Plant_Disease_Diagnosis/data
    unzip data.zip
    unzip train.zip
    unzip test.zip
    ```
4. Train `main.ipynb`

5. Submit 
`./submission_xxx.csv`

