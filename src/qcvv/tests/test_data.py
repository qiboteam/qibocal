# -*- coding: utf-8 -*-
"""Some tests for the Dataset class"""
import tempfile

import numpy as np
import pytest
from pint import DimensionalityError, UndefinedUnitError

from qcvv.data import Dataset


def random_dataset(length):
    data = Dataset()
    for _ in range(length):
        msr, i, q, phase = np.random.rand(len(data.df.columns))
        data.add({"MSR[V]": msr, "i[V]": i, "q[V]": q, "phase[deg]": phase})
    return data


def test_data_initialization():
    """Test Dataset constructor"""
    data = Dataset()
    assert len(data.df.columns) == 4
    assert list(data.df.columns) == ["MSR", "i", "q", "phase"]

    data1 = Dataset({"attenuation": "dB"})
    assert len(data1.df.columns) == 5
    assert list(data1.df.columns) == ["attenuation", "MSR", "i", "q", "phase"]


def test_units():
    """Test units of measure in Dataset"""
    data = Dataset()
    assert data.df.MSR.values.units == "volt"

    data1 = Dataset({"frequency": "Hz"})
    assert data1.df.frequency.values.units == "hertz"

    with pytest.raises(UndefinedUnitError):
        data2 = Dataset({"fake_unit": "fake"})


def test_add():
    """Test add method of Dataset"""
    data = random_dataset(5)
    assert len(data) == 5

    data1 = Dataset({"attenuation": "dB"})
    msr, i, q, phase, att = np.random.rand(len(data1.df.columns))
    data1.add(
        {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[deg]": phase,
            "attenuation[dB]": att,
        }
    )
    assert len(data1) == 1

    data2 = Dataset()
    msr, i, q, phase = np.random.rand(len(data2.df.columns))
    with pytest.raises(DimensionalityError):
        data2.add({"MSR[dB]": msr, "i[V]": i, "q[V]": q, "phase[deg]": phase})

    with pytest.raises(UndefinedUnitError):
        data2.add({"MSR[test]": msr, "i[V]": i, "q[V]": q, "phase[deg]": phase})
