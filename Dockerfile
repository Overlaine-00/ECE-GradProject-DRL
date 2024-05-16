FROM continuumio/miniconda3:latest AS base

WORKDIR /local
ARG DEBIAN_FRONTEND=noninteractive

# setting
RUN conda install python=3.11
RUN conda install -y numpy pandas scipy numba statsmodels matplotlib
RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
RUN conda install -c conda-forge jupyterlab ipykernel
RUN pip install scikit-learn


# SSH connection
ENV ROOTPW {YOUR_ROOT_PASSWORD}

RUN apt-get update && \
    apt-get install -y openssh-server && \
    apt-get clean && \
    mkdir /var/run/sshd

RUN ssh-keygen -A
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN echo 'root:${ROOTPW}' | chpasswd
RUN adduser root sudo

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]