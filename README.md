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

### Usage
1. `git clone https://github.com/glee1228/Plant-dacon.git`

2. Edit `docker-compose.yml`
    ```
    services:
      main:
        container_name: plant-dacon
        ...
        ports:
          - "{host ssh}:22"
        ipc: host
        stdin_open: true
    ```

3. `docker-compose up -d`

4. Download data from HDD to server docker container workspace path.
    ```
    #/workspace
    mkdir data
    cd data
    cp /mldisk/nfs_shared_/dh/plant_dacon/data.zip data.zip
    unzip data.zip
    unzip train.zip
    unzip test.zip
    ```
    
5. (option) Set password and Restart SSH for SFTP connection
    ```bash
    passwd
    /etc/init.d/ssh restart
    ```
    
6. Train  `baseline.py`
    ```bash
    #/workspace
    python baseline.py
    ```

7. Submit 
`{checkpoint directory}/submission.csv`
