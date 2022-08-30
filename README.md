# QCVV

This package provides Quantum Characterization Validation and Verification protocols using [Qibo](https://github.com/qiboteam/qibo) and [Qibolab](https://github.com/qiboteam/qibolab).

## Installation

The package can be installed by source:
```sh
git clone https://github.com/qiboteam/qcvv.git
cd qcvv
pip install .
```


### Developer instructions
For development make sure to install the package using [`poetry`](https://python-poetry.org/) and to install the pre-commit hooks:
```sh
git clone https://github.com/qiboteam/qcvv.git
cd qcvv
poetry install
pre-commit install
```

## Minimal working example
The command for executing calibration routines is the following:
```sh
qq <runcard>
```
where:
- `<runcard>`: yaml file containing the calibration routines to be performed. For more information see the documentation or the runcard examples in the `runcards` folder.

### Uploading reports to server

In order to upload the report to a centralized server, send to the server administrators your public ssh key (from the machine(s) you are planning to upload the report) and then use the `qq-upload <output_folder>` command. This program will upload your report to the server and generate an unique URL.
