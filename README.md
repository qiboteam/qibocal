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

## Development
- The documentation is available here: [![Documentation Status](https://readthedocs.org/projects/qibocal/badge/?version=latest)](https://qibocal.readthedocs.io/en/latest/)
- To build the documentation from source run
```sh
cd qibocal
poe docs
cd doc/build/html
open index.html
```
## Minimal working example

In order to test the installation check the qibocal version

```python
import qibocal

print(qibocal.__version__)
```
## Tests and benchmarks

To run the unit test you can use the command
```sh
poe test
```
## Contributing

Contribution, issues and feature request are welcome!
Feel free to check
![issues](https://img.shields.io/github/issues-closed/qiboteam/qibocal)
