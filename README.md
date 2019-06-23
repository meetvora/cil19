This package makes extensive use of GNU Make to simplify instructions.
Some instructions might not be supported on your system. Kindly edit `Makefile` accordingly.

## Setup
Run `make setup` to install all requirements, download data and make required directories.
Run `tree .` next to observe the changed directory structure.

## Train & Evaluate
Currently, the script trains a given model for given epochs and evaluates at the end.
Running `make train` starts the process, logs of which are flushed in the `log/` directory.

Experiments are named as "dd-mm--HH-min" denoting the day and time of start of execution.

##### Reading log
`tail -f log/<git-branch>/<exp-name>.log`

##### Configuring training parameters
`nano src/config.py`

##### Generating submission file
`make csv EXP=log/<git-branch>/<exp-name>_<model-name>`
