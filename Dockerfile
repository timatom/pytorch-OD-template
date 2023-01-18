FROM ubuntu:20.04

# Set timezone
ENV TZ=US/Eastern

# Install timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3, pip, and git
# RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.10 python3-pip && \
    apt-get install -y git

# Create symbolic link for python3
RUN ln -s $(which python3) /usr/local/bin/python

# Install PyTorch and Torchvision
RUN python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install OpenCV
RUN pip install opencv-python

# Make container directories and copy content from host
RUN mkdir /home/src
COPY ./src /home/src
