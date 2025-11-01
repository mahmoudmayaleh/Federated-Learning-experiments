FROM ubuntu:22.04
# FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt update && apt install -y openjdk-8-jdk stress iproute2 gettext-base python3-pip bc bash stress-ng
RUN pip3 install kafka-python confluent-kafka psutil numpy scikit-learn joblib scipy pandas ntplib

WORKDIR /data_sources
