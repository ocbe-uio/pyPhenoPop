FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
    && apt-get install -y \
    git \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip

RUN pip3 install -U --upgrade pip wheel \
    && pip3 install pyphenopop
