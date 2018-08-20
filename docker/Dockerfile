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
RUN conda create -n HPAT -c ehsantn -c numba -c anaconda -c conda-forge hpat

RUN echo "export PATH=/root/miniconda3/bin/:${PATH}" >> /root/.bashrc
RUN echo "source activate HPAT" >> /root/.bashrc

#ENV http_proxy ""
#ENV https_proxy ""
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
COPY cluster_key.pub /root/.ssh/
COPY cluster_key /root/.ssh/
RUN chmod 700 /root/.ssh/
RUN chmod 400 /root/.ssh/cluster_key
RUN chmod 700 /root/.ssh/cluster_key.pub

RUN echo "Host *" > /root/.ssh/config
RUN echo "    User root" >> /root/.ssh/config
RUN echo "    StrictHostKeyChecking no" >> /root/.ssh/config
RUN echo "    PreferredAuthentications publickey" >> /root/.ssh/config
RUN echo "    IdentityFile /root/.ssh/cluster_key" >> /root/.ssh/config

RUN echo -e "mpi_node1\nmpi_node2\nmpi_node3\nmpi_node4" > /root/hostfile

RUN cp /root/.ssh/cluster_key.pub /root/.ssh/authorized_keys

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
