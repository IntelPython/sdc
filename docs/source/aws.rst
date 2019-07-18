.. _aws:

AWS Setup
=========


This page describes a simple setup process for HPAT on Amazon EC2 instances. You need to have an account on Amazon Web Services (AWS)
and be familiar with the general AWS EC2 instance launch interface. The process below is for demonstration purposes only and is not
recommended for production usage due to security, performance and other considerations.

1. Launch instances
    a. Select a Linux instance type (e.g. Ubuntu Server 18.04, c5n types for high network bandwidth).
    b. Select number of instances (e.g. 4).
    c. Select placement group option for better network performance (check "add instance to placement group").
    d. Enable all ports in security group configuration to simplify MPI setup (add a new rule with "All traffic" Type and "Anywhere" Source).


2. Setup password-less ssh between instances
    a. Copy your key from your client to all instances. For example, on a Linux clients run this for
    all instances (find public host names from AWS portal)::

        scp -i "user.pem" user.pem ubuntu@ec2-11-111-11-111.us-east-2.compute.amazonaws.com:~/.ssh/id_rsa

    b. Disable ssh host key check by running this command on all instances::

        echo -e "Host *\n    StrictHostKeyChecking no" > .ssh/config

    c. Create a host file with list of private hostnames of instances on home directory of all instances::

        echo -e "ip-11-11-11-11.us-east-2.compute.internal\nip-11-11-11-12.us-east-2.compute.internal\n" > hosts

3. Install Anaconda Python distribution and HPAT on all instances::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n HPAT -c ehsantn -c anaconda -c conda-forge hpat
    source activate HPAT


4. Copy the `Pi example <https://github.com/IntelLabs/hpat#example>`_ to a file called pi.py in the home directory of
all instances and run it with and without MPI and see execution times.
You should see speed up when running on more cores ("-n 2" and "-n 4" cases)::

    python pi.py  # Execution time: 2.119
    mpiexec -f hosts -n 2 python pi.py  # Execution time: 1.0569
    mpiexec -f hosts -n 4 python pi.py  # Execution time: 0.5286


Possible next experiments from here are running a more complex example like the
`logistic regression example <https://github.com/IntelLabs/hpat/blob/master/examples/logistic_regression_rand.py>`_.
Furthermore, attaching a shared EFS storage volume and experimenting with parallel I/O in HPAT is recommended.
