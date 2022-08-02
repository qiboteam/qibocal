# -*- coding: utf-8 -*-
"""Some tests for the Dataset class"""
import pytest
from pint import UndefinedUnitError

from qcvv.data import Dataset


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
    # assert data.df.i.values.units == "volt"
    # assert data.df.q.values.units == "volt"
    # assert data.df.phase.values.units == "degree"

    data1 = Dataset({"frequency": "Hz"})
    assert data1.df.frequency.values.units == "hertz"

    with pytest.raises(UndefinedUnitError):
        data2 = Dataset({"fake_unit": "fake"})
