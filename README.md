# Qibocal
![Tests](https://github.com/qiboteam/qibocal/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibocal/branch/main/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/qiboteam/qibo)
[![Documentation Status](https://readthedocs.org/projects/qibocal/badge/?version=latest)](https://qibocal.readthedocs.io/en/latest/)

Qibocal provides Quantum Characterization Validation and Verification protocols using [Qibo](https://github.com/qiboteam/qibo) and [Qibolab](https://github.com/qiboteam/qibolab).

Qibocal key features:

- Automatization of calibration routines.

- Declarative inputs using runcard.

- Generation of a report.

## Installation

The package can be installed by source:
```sh
git clone https://github.com/qiboteam/qibocal.git
cd qibocal
pip install .
```


### Developer instructions
For development make sure to install the package using [`poetry`](https://python-poetry.org/) and to install the pre-commit hooks:
```sh
git clone https://github.com/qiboteam/qibocal.git
cd qibocal
poetry install
pre-commit install
```

## Minimal working example
### Calibrate with a runcard
To run all the calibration routines specified in the ```runcard```, Qibocal uses the command 
```sh
qq <runcard> -o <output_folder>
```
if ```<output_folder>``` is specified, the results will be saved in it, otherwise ```qq``` will automatically create a default folder containing the current date and the username.


### Uploading reports to server

In order to upload the report to a centralized server, send to the server administrators your public ssh key (from the machine(s) you are planning to upload the report) and then use the `qq-upload <output_folder>` command. This program will upload your report to the server and generate an unique URL.

## Contributing

Contribution, issues and feature request are welcome!
Feel free to check
<a href="https://github.com/qiboteam/qibocal/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues-closed/qiboteam/qibocal"/></a>
