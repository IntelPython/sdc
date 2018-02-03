FROM ubuntu:16.04

WORKDIR /root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

#ENV http_proxy http://proxy-chain.intel.com:911
#ENV https_proxy http://proxy-chain.intel.com:911

RUN apt-get update
RUN apt-get install -y wget bzip2 git vim
RUN wget -q https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
ENV PATH="/root/anaconda3/bin/:${PATH}"

RUN conda create -y -n HPAT
RUN source activate HPAT && \
    conda install -y numpy scipy pandas python=3.6 &&\
    conda install -y pyarrow=0.8.* mpich -c conda-forge &&\
    conda install -y hpat -c ehsantn

RUN echo "export PATH=/root/anaconda3/bin/:${PATH}" >> /root/.bashrc
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
