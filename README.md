# Qibocal
[![codecov](https://codecov.io/gh/qiboteam/qibocal/branch/main/graph/badge.svg?token=1EKZKVEVX0)](https://codecov.io/gh/qiboteam/qibocal)
![PyPI - Version](https://img.shields.io/pypi/v/qibocal)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qibocal)

Qibocal provides calibration protocols using [Qibo](https://github.com/qiboteam/qibo) and [Qibolab](https://github.com/qiboteam/qibolab).

Qibocal key features:

- Declarative inputs using runcard.

- Generation of a report.

## Documentation

[![docs](https://github.com/qiboteam/qibocal/actions/workflows/publish.yml/badge.svg)](https://qibo.science/qibocal/stable/)

Qibocal documentation is available [here](https://qibo.science/qibocal/stable/).

>[!NOTE]
> Qibocal `main` contains some breaking changes compared to `0.1` versions.
> A small guide to make the transition as smooth as possible can be found [`here`](changes.md).

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

Here is an example on how to run a Rabi experiment in Qibocal.

```py
from qibocal import create_calibration_platform
from qibocal.protocols import rabi_amplitude

# create platform
platform = create_calibration_platform("qubit")

# define qubits where the protocols will be executed
targets = [0]

# define protocol parameters
params = rabi_amplitude.parameters_type.load(dict(
        min_amp=0.01,
        max_amp=0.2,
        step_amp=0.02,
        nshots=2000,
        pulse_length=40,
        ))

# acquire
data, acquisition_time = rabi_amplitude.acquisition(
                                                    params=params,
                                                    platform=platform,
                                                    targets=targets
                                                    )

# post-processing
results, fit_time = rabi_amplitude.fit(data=data)

# visualize the results
plots, table = rabi_amplitude.report(data=data, results=results, target=target[0])
plots[0].show()
```

<p align="center">
  <img alt="Rabi" src="doc/source/img/rabi.png" >
</p>

The table is written in HTML and can be visualized in Python with

```py
from IPython import display
display.HTML(table)
```
<p align="center">
<table style="width: 70%; border-collapse: collapse; text-align: center; margin: 40px auto 0 auto; font-family: system-ui, sans-serif; font-size: 0.8em; border-radius: 15px;">
  <thead>
    <tr style="background-color: #f0e6ff;">
      <th style="padding: 8px;">Qubit</th>
      <th style="padding: 8px;">Parameters</th>
      <th style="padding: 8px;">Values</th>
      <th style="padding: 8px;">Errors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">0</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">Pi pulse amplitude [a.u.]</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">1.271e-1</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">0.002e-1</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">0</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">Pi pulse length [ns]</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">4.0e1</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">0e1</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">0</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">chi² reduced</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">9.0e-1</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">3e-1</td>
    </tr>
  </tbody>
</table>
</p>

The same experiment can also be run using the following yaml file

```yaml
platform: qubit

targets: [0]

- id: rabi
  operation: rabi_amplitude
  parameters:
    min_amp: 0.01
    max_amp: 0.2
    step_amp: 0.02
    nshots: 2000
    pulse_length: 40

```
### How to run protocols
To run the protocol Qibocal uses the `qq run` command
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
