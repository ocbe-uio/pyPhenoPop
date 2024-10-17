FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
    && apt-get install -y \
    git \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    && apt-get clean

RUN pip3 install --no-cache-dir -U --upgrade pip wheel \
    && pip3 install --no-cache-dir pyphenopop
