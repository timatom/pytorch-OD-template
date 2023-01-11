FROM ubuntu:20.04

# Install python3, pip, and git
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y git

# Install PyTorch and Torchvision
RUN python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install OpenCV
RUN pip install opencv-python

# Make container directories and copy content from host
RUN mkdir /home/src
COPY ./src /home/src
