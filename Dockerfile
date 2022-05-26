FROM  pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# For other versions of the apt update command except version 18.04
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && apt-get -y install \
    apt-utils git vim openssh-server


RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN pip install --upgrade pip
RUN pip install setuptools
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR /workspace
ADD . .
ENV PYTHONPATH $PYTHONPATH:/workspace

RUN pip install -r requirements.txt
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install unzip

RUN chmod -R a+w /workspace
RUN /bin/bash


