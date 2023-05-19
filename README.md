# Qibocal
![Tests](https://github.com/qiboteam/qibocal/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibocal/branch/main/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/qiboteam/qibocal)
[![Documentation Status](https://readthedocs.org/projects/qibocal/badge/?version=latest)](https://qibocal.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7662185.svg)](https://doi.org/10.5281/zenodo.7662185)

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

This section shows the steps to perform a resonator spectroscopy with Qibocal.
### Write a runcard
A runcard contains all the essential information to run a specific task.
For our purposes, we can use the following:
```yml
platform: tii1q

qubits: [0]

format: csv

actions:
   resonator_spectroscopy:
     lowres_width: 5_000_000
     lowres_step: 2_000_000
     highres_width: 1_500_000
     highres_step: 200_000
     precision_width: 1_500_000
     precision_step: 100_000
     software_averages: 1
     points: 5
```
### Run the routine
To run all the calibration routines specified in the ```runcard```, Qibocal uses the `qq` command
```sh
qq <runcard> -o <output_folder>
```
if ```<output_folder>``` is specified, the results will be saved in it, otherwise ```qq``` will automatically create a default folder containing the current date and the username.

### Visualize the data

Qibocal gives the possibility to live-plotting with the `qq-live` command
```sh
qq-live <output_folder>
```
### Uploading reports to server

In order to upload the report to a centralized server, send to the server administrators your public ssh key (from the machine(s) you are planning to upload the report) and then use the `qq-upload <output_folder>` command. This program will upload your report to the server and generate an unique URL.

## Contributing

Contributions, issues and feature requests are welcome!
Feel free to check
<a href="https://github.com/qiboteam/qibocal/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues-closed/qiboteam/qibocal"/></a>
