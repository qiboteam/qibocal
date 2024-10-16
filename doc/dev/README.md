# Qibocal Roadmap

## Protocols
### <font color="green">High Priority</font>
#### Single qubit

- Rabi and flipping to calibrate pi/2 rotations (https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf par. 5.9)
- Kernels for integration
- Drag with detuning  (https://arxiv.org/pdf/1904.06560)
- Cross-entropy benchmarking
- SNR in spectroscopies and SNR in the IQ plane
  https://dsp.stackexchange.com/questions/24372/what-is-the-connection-between-analog-signal-to-noise-ratio-and-signal-to-noise.
  https://arxiv.org/pdf/2106.06173.pdf
- Improve readout amplitude optimization with the outliers probability
(https://escholarship.org/content/qt0g29b4p0/qt0g29b4p0.pdf?t=prk0gj)
- Cryoscope (https://arxiv.org/pdf/1907.04818, https://github.com/qua-platform/qua-libs/blob/main/Quantum-Control-Applications/Superconducting/Single-Flux-Tunable-Transmon/Use%20Case%201%20-%20Paraoanu%20Lab%20-%20Cryoscope/readme.md)

#### Two qubits

- Cross resonance gates (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.134507)
- SNZ / Martini Ansatz (https://arxiv.org/pdf/2008.07411 https://arxiv.org/pdf/1402.5467)
- Improve and test coupler routines
- Measure ZZ coupling in couplers (flux amplitude vs coupling) (Manenti Motta, par. 14.8.4)
- Improve and test iSWAP implementation

### <font color="green">Low Priority</font>
#### Single Qubit

- Calibrate the other qubit states
- Carr-Purcell-Meiboom_Gill sequence
- Explore cosine pulse as X pulse (https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf par. 5.4)
- Active reset
- Measurement tomography https://arxiv.org/pdf/1310.6448.pdf
- XY-Z timing
(https://escholarship.org/content/qt0g29b4p0/qt0g29b4p0.pdf?t=prk0gj par. 5.10)
- Optimal control with randomize benchmarking (https://arxiv.org/pdf/1403.0035)
- Quantum volume
- Gate Set Tomography
