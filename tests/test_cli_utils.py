"""Testing cli utils."""

from qibolab import create_platform

from qibocal.cli.utils import create_qubits_dict

PLATFORM = create_platform("dummy")


def test_create_qubits_dict():
    """Testing create_qubits_dict."""
    qubits = list(PLATFORM.qubits)

    target_qubits = create_qubits_dict(qubits, PLATFORM)
    assert target_qubits == PLATFORM.qubits

    pairs = [list(pair) for pair in list(PLATFORM.pairs)]
    target_pairs = create_qubits_dict(pairs, PLATFORM)
    assert target_pairs == {pair: PLATFORM.pairs[pair] for pair in PLATFORM.pairs}
    assert [0, 1, 2] == create_qubits_dict([0, 1, 2], None)
