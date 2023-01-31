FROM ubuntu:20.04

# Defining versions for python and libraries
ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1
ARG CUDA_VERSION=cu117

# Set timezone
ENV TZ=US/Eastern

# Install timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND noninteractive

# Install python3, pip, and git
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get --no-install-recommends install -y \
        build-essential \
        ca-certificates \
        cmake \
        cmake-data \
        pkg-config \
        libcurl4 \
        libsm6 \
        libxext6 \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        unzip \
        curl \
        wget \
        python${PYTHON_VERSION} \
        # python${PYTHON_VERSION}-dev \
        # python${PYTHON_VERSION}-distutils \
        ffmpeg \
    && apt-get install -y git

# TODO: Install Voxel51 dependencies
# Note: Refer to the following links for more information
#       1. https://github.com/voxel51/fiftyone/blob/develop/Dockerfile
#       2. https://github.com/aegean-ai/fiftyone-examples
#       3. https://docs.voxel51.com/getting_started/install.html

# Create symbolic link for python3
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -s /usr/local/lib/python${PYTHON_VERSION} /usr/local/lib/python

# Install and update pip
# TODO: Figure out why I'm getting "pip module not found" error!!!
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip

# Install PyTorch and Torchvision
RUN pip install torch==${PYTORCH_VERSION}+${CUDA_VERSION} torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION} -f https://download.pytorch.org/whl/torch_stable.html

# Install OpenCV
RUN pip install opencv-python

# Install evaluation tools
RUN pip --no-cache-dir install fiftyone pycocotools

# Install other libraries
RUN pip install tqdm

# Make container directories and copy content from host
RUN mkdir /home/src
COPY ./src /home/src
