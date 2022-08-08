# QCVV

This package provides Quantum Characterization Validation and Verification protocols using [Qibo](https://github.com/qiboteam/qibo) and [Qibolab](https://github.com/qiboteam/qibolab).

## Installation

The package can be installed by source:
```sh
git clone https://github.com/qiboteam/qcvv.git
cd qcvv
pip install .
```

To be able to connect to the `TII` quantum hardware remember to install `qcvv` with the tiiq extension:
```sh
pip install .[tiiq]
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
