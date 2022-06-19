#!/bin/bash
# Name: cluster_setup.sh
# Purpose: Set up the servers for experimenting with DiBB
# --------------------------------------------------------------------------

# ------------------------------ Instructions ------------------------------
#
#  Before you run this script, make sure to clone dibb and coco. This script 
#  assumes that you have access to all of the cluster nodes and that they are
#  saved in your ssh config file with a certain naming pattern. The user to
#  which the script connects needs to have password-less sudo rights in order
#  to install the required software. Make sure that the variable $LOCAL_EXP_DIR
#  points to the directory that includes the Pipenv files and that the variable
#  $REMOTE_USERNAME is changed if your remote user is not called ubuntu.
#
#  Requirements: GNU Parallel
#
# --------------------------------------------------------------------------


# ----------------------------- Setup Variables ----------------------------

MACHINE_PATTERN='machine_[0-9]+'  # Chooses the machines that are used based reges pattern in .ssh/config
LOCAL_EXP_DIR='~/dibb_coco/'      # Path to the local experiment directory
REMOTE_USERNAME='ubuntu'

# Get all the hostnames corresponding to the regex pattern
MACHINES=$(cat ~/.ssh/config | egrep -o "($MACHINE_PATTERN)" | tr '\n' ',')

# You can use these lines to override the MACHINES variable
# MACHINES="machine_1,machine_2,machine_3,"

NUM_MACHINES=$(tr -dc ',' <<< "$MACHINES" | awk "{ print length; }")
MACHINES=${MACHINES:0:-1}  # remove last comma
MACHINES_RSYNC_ARGS=$(echo "$MACHINES" | tr ',' ' ')
JOBS_PARAM="-j$NUM_MACHINES"  # Specify number of jobs (= number of machines)


# ------------------------------ Script start ------------------------------

# Install software
parallel --onall $JOBS_PARAM -X --tty -S $MACHINES "echo {} | sudo locale-gen en_US.UTF-8 de_CH.UTF-8; sudo dpkg-reconfigure --frontend noninteractive locales"
parallel --onall $JOBS_PARAM -X --tty -S $MACHINES "echo {} | sudo apt install python3-pip build-essential python-dev python-setuptools -y"

# Upload local coco folder in home directory to each machine in the cluster
parallel $JOBS_PARAM -X -N1 rsync -azhP ~/coco/ --exclude '.git' {}:~/coco/ ::: $MACHINES_RSYNC_ARGS

# COCO setup
parallel --onall $JOBS_PARAM -X -S $MACHINES "{}" ::: <<- EndCommands
  pip3 install pipenv
  cd ~/coco
  python3 -m pipenv --two install numpy scipy matplotlib six
  python3 -m pipenv run python ~/coco/do.py run-python
EndCommands

# Copy DiBB and the experiment pipenv files
parallel $JOBS_PARAM -X -N1 rsync -azhP ~/dibb/ --exclude '.git' {}:~/dibb/ ::: $MACHINES_RSYNC_ARGS
parallel $JOBS_PARAM -X -N1 rsync -azhP "${LOCAL_EXP_DIR}Pipfile* {}:~/ ::: $MACHINES_RSYNC_ARGS

# DiBB and pipenv setup
parallel --onall $JOBS_PARAM -X -S $MACHINES "{}" ::: <<- EndCommands
  cd
  sed -i "s/ubuntu/$REMOTE_USERNAME/g" Pipfile
  sed -i "s/ubuntu/$REMOTE_USERNAME/g" Pipfile.lock
  python3 -m pipenv --three sync
EndCommands