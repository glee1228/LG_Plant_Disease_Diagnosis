# Plant-dacon
Plant Disease Classification on Multi-modal Features in DACON.


### Directory structure
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

### Docker Container Usage
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
    ```
    # ~/LG_Plant_Disease_Diagnosis
    mkdir data
    cd data
    (Download data to ~/LG_Plant_Disease_Diagnosis/data/)
    ```

4. Build docker image clearly and create containers
    ```
    docker-compose build --no-cache
    docker-compose up -d
    docker attach plant-lg-dacon
    ```
    
5. Unzip train, test data
    ```
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
