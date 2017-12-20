FROM ubuntu:16.04

WORKDIR /root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ENV http_proxy http://proxy-chain.intel.com:911
ENV https_proxy http://proxy-chain.intel.com:911

RUN apt-get update
RUN apt-get install -y wget bzip2 llvm-4.0 git vim
RUN wget -q https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
ENV PATH="/root/anaconda3/bin/:${PATH}"

RUN conda create -y -n HPAT
RUN source activate HPAT && \
    conda install -y numpy scipy pandas llvmlite=0.20 python=3.6 &&\
    conda install pyarrow=0.8.* mpich -c conda-forge

RUN echo "export PATH=/root/anaconda3/bin/:${PATH}" >> /root/.bashrc
RUN echo "source activate HPAT" >> /root/.bashrc
RUN echo -e "channels:\n  - ehsantn\n  - defaults\n  - conda-forge\n  - intel\n\nanaconda_upload: True\n" > /root/.condarc
# RUN git clone https://github.com/IntelLabs/hpat.git
# RUN git clone https://github.com/IntelLabs/numba.git &&\
#     cd numba && git checkout hpat_req && cd ..
