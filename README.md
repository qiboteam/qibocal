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
Make sure to first install [Qibolab](https://github.com/qiboteam/qibolab) in the same environment where you installed `qcvv`.

The command for executing calibration routines is the following:
```sh
qq <runcard> <output_folder>
```
where:
- `<runcard>`: yaml file containing the calibration routines to be performed. For more information see the documentation or the runcard examples in the `runcards` folder.
- `<output_folder>`: the folder that will be created to store the output, in order to use an existing folder you can use the option `--force`.
