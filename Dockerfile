FROM ubuntu:16.04   

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

#Runit
RUN apt-get install -y runit 
CMD /usr/sbin/runsvdir-start

#SSHD
RUN apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd && \
    echo 'root:root' |chpasswd
RUN sed -i "s/session.*required.*pam_loginuid.so/#session    required     pam_loginuid.so/" /etc/pam.d/sshd
RUN sed -i "s/PermitRootLogin without-password/#PermitRootLogin without-password/" /etc/ssh/sshd_config

#Utilities
RUN apt-get install -y vim less net-tools inetutils-ping curl git telnet nmap socat dnsutils netcat tree htop unzip sudo software-properties-common

#Required by Python packages
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python-dev python-pip liblapack-dev libatlas-base-dev gfortran libfreetype6 libfreetype6-dev libpng12-dev python-lxml libyaml-dev g++ libffi-dev

#0MQ
RUN cd /tmp && \
    wget http://download.zeromq.org/zeromq-4.0.3.tar.gz && \
    tar xvfz zeromq-4.0.3.tar.gz && \
    cd zeromq-4.0.3 && \
    ./configure && \
    make install && \
    ldconfig

#Upgrade pip
#RUN pip install -U setuptools
RUN pip install -U pip
RUN pip install --upgrade pip


#matplotlib needs latest distribute
RUN pip install -U distribute

#IPython
RUN pip install ipython
ENV IPYTHONDIR /ipython
RUN mkdir /ipython && \
    ipython profile create nbserver

#NumPy
RUN pip install numpy

#Pandas
RUN pip install pandas

#Optional
RUN pip install scipy
RUN apt-get install pkg-config
RUN pip install matplotlib

#Add runit services
#ADD sv /etc/services

#Need to install acor for carma_pack
RUN pip install acor


RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential wget libbz2-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ARG boost_version=1.64.0
ARG boost_dir=boost_1_64_0
ENV boost_version ${boost_version}

RUN wget https://dl.bintray.com/boostorg/release/${boost_version}/source/${boost_dir}.tar.gz \
    && tar xfz ${boost_dir}.tar.gz \
    && rm ${boost_dir}.tar.gz \
    && cd ${boost_dir} \
    && ./bootstrap.sh \
    && ./b2 --prefix=/usr -j 4 link=shared runtime-link=shared install \
    && cd .. && rm -rf ${boost_dir} && ldconfig

RUN apt-get update

CMD bash

#install armadillo libraries for carma_pack
RUN apt-get install -y libarmadillo-dev

RUN apt-get install -y libboost-python-dev

#set environment variales for carma_pack install
ENV BOOST_DIR=/usr/include/boost
ENV ARMADILLO_DIR=/usr/lib
ENV NUMPY_DIR=/usr/local/lib/python2.7/dist-packages

RUN apt-get install -y git
RUN apt-get install -y python-tk

ENV CARMA_DIR=/carma/carma_pack

RUN mkdir /carma && git clone https://github.com/brandonckelly/carma_pack.git $CARMA_DIR && cd ${CARMA_DIR}/src && ls && echo 'next ls' && ls .. && python setup.py install && ldconfig

RUN pip install astroML
RUN pip install astroML_addons
RUN pip install pyfits

RUN apt-get install -qqy x11-apps
ENV DISPLAY :0
