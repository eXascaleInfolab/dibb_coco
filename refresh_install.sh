#!/bin/bash

# Refresh the experiment environment on the Ray cluster machines

IPS="134.21.220.201 134.21.220.202 134.21.220.203 134.21.220.205"
DIR="dibb_coco"
USER="ubuntu"

function one_server() {
  rsync -azhP ${HOME}/${DIR} ${USER}@${1}:/home/${USER}/ --exclude='.git/'
  ssh ${USER}@${1} "\
    rm -f Pipfile;\
    python3 -m pipenv --rm;\
    cd ${DIR};\
    python3 -m pipenv --rm;\
    python3 -m pipenv sync;"
}

for ip in ${IPS}; do
  one_server $ip &
done
