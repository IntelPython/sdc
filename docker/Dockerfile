FROM ubuntu:16.04

WORKDIR /root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update
RUN apt-get install -y wget bzip2 llvm-4.0 make libc6-dev gcc-4.9 g++-4.9 git
RUN wget -q https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
RUN bash Anaconda3-4.4.0-Linux-x86_64.sh -b
ENV PATH="/root/anaconda3/bin/:${PATH}"

ENV CC gcc-4.9
ENV CXX g++-4.9
ENV LD gcc-4.9
ENV LDSHARED "gcc-4.9 -shared"

RUN conda create -y -n HPAT
RUN source activate HPAT && \
    conda install -y numpy scipy pandas mpich2 llvmlite &&\
    git clone https://github.com/IntelLabs/numba.git && \
    cd numba && git checkout hpat_req && python setup.py install && cd .. &&\
    git clone https://github.com/IntelLabs/hpat.git && \
    cd hpat && LDSHARED="mpicxx -shared -cxx=g++-4.9" \
    CC="mpicxx -std=c++11 -cxx=g++-4.9" CXX="mpicxx -std=c++11 -cxx=g++-4.9" \
    python setup.py install

RUN echo "source activate HPAT" >> /root/.bashrc
