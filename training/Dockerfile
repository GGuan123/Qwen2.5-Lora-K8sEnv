FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
ENV HF_ENDPOINT=https://hf-mirror.com

RUN apt update && apt install -y python3 python3-pip git && \
    ln -sf python3 /usr/bin/python && \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r /tmp/requirements.txt

COPY train.py /app/train.py
WORKDIR /app

ENTRYPOINT ["python3", "train.py"]