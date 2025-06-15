#FROM ubuntu:latest
#LABEL authors="nahyeon"
#
#ENTRYPOINT ["top", "-b"]

# Python 3.10 기반 slim 이미지 사용
FROM python:3.10-slim

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /Machine-Learning

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 소스코드 전체 복사
COPY . .