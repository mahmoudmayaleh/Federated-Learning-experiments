FROM python:3.10.17-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

RUN apt update && apt install -y curl && pip install kafka-python confluent-kafka numpy scikit-learn ntplib
COPY ./NDPPF/network_data_preprocessing_function /network_data_preprocessing_function

WORKDIR /network_data_preprocessing_function

