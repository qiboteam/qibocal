# Qibocal
[![codecov](https://codecov.io/gh/qiboteam/qibocal/branch/main/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/qiboteam/qibocal)
![PyPI - Version](https://img.shields.io/pypi/v/qibocal)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qibocal)

Qibocal provides Quantum Characterization Validation and Verification protocols using [Qibo](https://github.com/qiboteam/qibo) and [Qibolab](https://github.com/qiboteam/qibolab).

Qibocal key features:

- Declarative inputs using runcard.

- Generation of a report.

## Documentation

[![docs](https://github.com/qiboteam/qibocal/actions/workflows/publish.yml/badge.svg)](https://qibo.science/qibocal/stable/)

Qibocal documentation is available [here](https://qibo.science/qibocal/stable/).

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

targets: [0]

- id: resonator spectroscopy high power
  operation: resonator_spectroscopy
  parameters:
    freq_width: 10_000_000
    freq_step: 500_000
    amplitude: 0.4
    power_level: high
    nshots: 1024
    relaxation_time: 5_000

```
### How to run protocols
To run the protocols specified in the ```runcard```, Qibocal uses the `qq run` command
```sh
qq run <runcard> -o <output_folder>
```
if ```<output_folder>``` is specified, the results will be saved in it, otherwise ```qq``` will automatically create a default folder containing the current date and the username.


### Uploading reports to server

In order to upload the report to a centralized server, send to the server administrators your public ssh key (from the machine(s) you are planning to upload the report) and then use the `qq upload <output_folder>` command. This program will upload your report to the server and generate an unique URL.

## Contributing

Contributions, issues and feature requests are welcome!
Feel free to check
<a href="https://github.com/qiboteam/qibocal/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues-closed/qiboteam/qibocal"/></a>

## Citation policy
[![arXiv](https://img.shields.io/badge/arXiv-2303.10397-b31b1b.svg)](https://arxiv.org/abs/2303.10397)
[![DOI](https://zenodo.org/badge/511836317.svg)](https://zenodo.org/badge/latestdoi/511836317)


If you use the package please refer to [the documentation](https://qibo.science/qibo/stable/appendix/citing-qibo.html#publications) for citation instructions

## Ongoing development

A non-exhaustive list of possible protocols to be implemented in Qibocal is collected
[here](doc/dev/README.md).
