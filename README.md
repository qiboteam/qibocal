# Qibocal
![Tests](https://github.com/qiboteam/qibocal/workflows/Tests/badge.svg)
[![codecov](https://codecov.io/gh/qiboteam/qibocal/branch/main/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/qiboteam/qibocal)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7662185.svg)](https://doi.org/10.5281/zenodo.7662185)

Qibocal provides Quantum Characterization Validation and Verification protocols using [Qibo](https://github.com/qiboteam/qibo) and [Qibolab](https://github.com/qiboteam/qibolab).

Qibocal key features:

- Automatization of calibration protocols.

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

- id: resonator spectroscopy high power
  priority: 0
  operation: resonator_spectroscopy
  parameters:
    freq_width: 10_000_000
    freq_step: 500_000
    amplitude: 0.4
    power_level: high
    nshots: 1024
    relaxation_time: 0

```
### How to run protocols
To run the protocols specified in the ```runcard```, Qibocal uses the `qq auto` command
```sh
qq auto <runcard> -o <output_folder>
```
if ```<output_folder>``` is specified, the results will be saved in it, otherwise ```qq``` will automatically create a default folder containing the current date and the username.


### Uploading reports to server

In order to upload the report to a centralized server, send to the server administrators your public ssh key (from the machine(s) you are planning to upload the report) and then use the `qq upload <output_folder>` command. This program will upload your report to the server and generate an unique URL.

## Contributing

Contributions, issues and feature requests are welcome!
Feel free to check
<a href="https://github.com/qiboteam/qibocal/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues-closed/qiboteam/qibocal"/></a>
