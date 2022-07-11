# DiBB experiments with COCO BBOB and BBOB-Largescale

## Experiment template
This repository contains the code that I used in my thesis to run the COCO benchmark with DiBB. The file `experiment_template.py` is used to start one experiment (or a batch of them). Each experiment is characterized by different parameters such as a suite (bbob or bbob-largescale), a dimension and a number of blocks. If you want to run the experiments yourself, make sure that you have enough machines, i.e. number of machines >= number of blocks + 1. To start multiple experiments at once, I suggest making a copy of the template file and running it individually. If you configured something wrong, you can then just stop one of the experiments without affecting the others. 

## Merging the experiment data
After executing an experiment, the experiment data will be stored on each node of the cluster that hosted a BlockWorker. More specifically, the data will be stored in ~/exdata/{script_name}/{experiment_name}, where {script_name} is the name of the python file without the extension, and {experiment_name} is the name of the experiment specified in the experiment file. To merge the experiment data, you should first download all of it to your local machine. You may want to use this bash function to do so (install GNU Parallel first):

```bash
download_experiment_data() {
  MACHINE_PATTERN='machine_[0-9]+'  # Chooses the machines that are used based on .ssh/config and regex pattern
  MACHINES=$(cat ~/.ssh/config | egrep -o "($MACHINE_PATTERN)" | tr '\n' ',')

  # You can use these lines to override the MACHINES variable
  # MACHINES="machine_01,machine_02,machine_03,"

  NUM_MACHINES=$(tr -dc ',' <<< "$MACHINES" | awk "{ print length; }")
  MACHINES=${MACHINES:0:-1}  # remove last comma
  MACHINES_RSYNC_ARGS=$(echo "$MACHINES" | tr ',' ' ')
  JOBS_PARAM="-j$NUM_MACHINES"  # Specify number of jobs (= number of machines)

  # Add trailing slash to path if not already there
  if [[ "${2: -1}" == '/' ]]; then
    path=$2
  else
    path="${2}/"
  fi

  echo 'Downloading experiment data from cluster...'
  echo 'This might take a while'
  parallel $JOBS_PARAM -X -N1 rsync -azhP --ignore-existing {}:~/exdata/$1/ "${path}${1}/" ::: $MACHINES_RSYNC_ARGS
  echo "Experiments downloaded to $path$1/"
}
```

Once all of the data has been downloaded, you can use the script `merging.py` to merge the experiment data. This might take a while. Make sure to specify the correct paths at the end of the file.

## Acknowledgements

This code is part of my master's thesis and has also been used in the experiments of the [resulting paper](https://exascale.info/assets/pdf/cuccu2022gecco.pdf), which has been published at GECCO 2022, The 24th Genetic and Evolutionary Computation Conference. You can use the following bibtex entry to cite the paper:

```bibtex
@inproceedings{cuccu2022gecco,
  author    = {Cuccu, Giuseppe and Rolshoven, Luca and Vorpe, Fabien and Cudr\'{e}-Mauroux, Philippe and Glasmachers, Tobias},
  title     = {{DiBB}: Distributing Black-Box Optimization},
  year      = {2022},
  isbn      = {9781450392372},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3512290.3528764},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  pages     = {341–349},
  numpages  = {9},
  keywords  = {parallelization, neuroevolution, evolution strategies, distributed algorithms, black-box optimization},
  location  = {Boston, Massachusetts},
  series    = {GECCO '22},
  url       = {https://exascale.info/assets/pdf/cuccu2022gecco.pdf}
}
```
