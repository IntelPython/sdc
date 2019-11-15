FROM ubuntu:16.04

WORKDIR /root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

#ENV http_proxy http://proxy-chain.intel.com:911
#ENV https_proxy http://proxy-chain.intel.com:911

RUN apt-get update && apt-get install -y wget bzip2 git vim make
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN chmod +x miniconda.sh
RUN ./miniconda.sh -b
ENV PATH="/root/miniconda3/bin/:${PATH}"
RUN echo "export PATH=/root/miniconda3/bin/:${PATH}" >> /root/.bashrc
RUN conda install -y conda-build anaconda-client conda-verify
#RUN conda create -y -n BUILD python=3.6


#RUN echo "source activate BUILD" >> /root/.bashrc
RUN echo -e "channels:\n  - ehsantn\n  - defaults\n  - conda-forge\n\nanaconda_upload: False\n" > /root/.condarc
#RUN git clone https://github.com/IntelLabs/hpat.git
#RUN conda install