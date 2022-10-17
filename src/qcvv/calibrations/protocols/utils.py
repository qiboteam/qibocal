from numpy import sqrt, pi
# To not define the parameters for one qubit Cliffords every time a
# new qubits is drawn define the parameters as global variable.
# This are parameters for all 24 one qubit clifford gates.
onequbit_clifford_params = [
    (0, 0, 0, 0), (pi, 1, 0, 0), (pi,0, 1, 0),
    (pi, 0, 0, 1), (pi/2, 1, 0, 0), (-pi/2, 1, 0, 0),
    (pi/2, 0, 1, 0), (-pi/2, 0, 1, 0), (pi/2, 0, 0, 1),
    (-pi/2, 0, 0, 1),
    (pi, 1/sqrt(2), 1/sqrt(2), 0),
    (pi, 1/sqrt(2), 0, 1/sqrt(2)),
    (pi, 0, 1/sqrt(2), 1/sqrt(2)),
    (pi, -1/sqrt(2), 1/sqrt(2), 0),
    (pi, 1/sqrt(2), 0, -1/sqrt(2)),
    (pi, 0, -1/sqrt(2), 1/sqrt(2)),
    (2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (-2*pi/3, 1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (-2*pi/3, -1/sqrt(3), 1/sqrt(3), 1/sqrt(3)),
    (2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    (-2*pi/3, 1/sqrt(3), -1/sqrt(3), 1/sqrt(3)),
    (2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3)),
    (-2*pi/3, 1/sqrt(3), 1/sqrt(3), -1/sqrt(3))]