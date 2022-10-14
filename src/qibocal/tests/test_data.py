# -*- coding: utf-8 -*-
"""Some tests for the Dataset class"""
import numpy as np
import pytest
from pint import DimensionalityError, UndefinedUnitError

from qibocal.data import Data, Dataset


def random_dataset(length, options=None):
    data = Dataset(options=options)
    for l in range(length):
        msr, i, q, phase = np.random.rand(4)
        pulse_sequence_result = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[deg]": phase,
        }
        add_options = {}
        if options is not None:
            for option in options:
                add_options[option] = str(l)
        data.add({**pulse_sequence_result, **add_options})

    return data


def random_data(length):
    data = Data()
    for i in range(length):
        data.add({"int": int(i), "float": float(i), "string": str(i), "bool": bool(i)})
    return data


def test_data_initialization():
    """Test Dataset constructor"""
    data = Dataset()
    assert len(data.data.columns) == 4
    assert list(data.data.columns) == ["MSR", "i", "q", "phase"]

    data1 = Dataset(quantities={"attenuation": "dB"})
    assert len(data1.data.columns) == 5
    assert list(data1.data.columns) == ["attenuation", "MSR", "i", "q", "phase"]

    data2 = Dataset(quantities={"attenuation": "dB"}, options=["option1"])
    assert len(data2.data.columns) == 6
    assert list(data2.data.columns) == [
        "option1",
        "attenuation",
        "MSR",
        "i",
        "q",
        "phase",
    ]


def test_dataset_units():
    """Test units of measure in Dataset"""
    dataset = Dataset()
    assert dataset.data.MSR.values.units == "volt"

    dataset1 = Dataset(quantities={"frequency": "Hz"})
    assert dataset1.data.frequency.values.units == "hertz"

    with pytest.raises(UndefinedUnitError):
        dataset2 = Dataset(quantities={"fake_unit": "fake"})


def test_dataset_add():
    """Test add method of Dataset"""
    dataset = random_dataset(5)
    assert len(dataset) == 5

    dataset1 = Dataset(quantities={"attenuation": "dB"})
    msr, i, q, phase, att = np.random.rand(len(dataset1.data.columns))
    dataset1.add(
        {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[deg]": phase,
            "attenuation[dB]": att,
        }
    )
    assert len(dataset1) == 1

    dataset1.add(
        {
            "MSR[V]": 0,
            "i[V]": 0.0,
            "q[V]": 0.0,
            "phase[deg]": 0,
            "attenuation[dB]": 1,
        }
    )
    assert len(dataset1) == 2

    dataset2 = Dataset()
    msr, i, q, phase = np.random.rand(len(dataset2.data.columns))
    with pytest.raises(DimensionalityError):
        dataset2.add({"MSR[dB]": msr, "i[V]": i, "q[V]": q, "phase[deg]": phase})

    with pytest.raises(UndefinedUnitError):
        dataset2.add({"MSR[test]": msr, "i[V]": i, "q[V]": q, "phase[deg]": phase})

    dataset3 = random_dataset(10, options=["test"])
    assert len(dataset3) == 10


def test_data_add():
    """Test add method of Data class"""
    data = random_data(5)
    assert len(data) == 5
    data.add({"int": 123, "float": 123.456, "string": "123", "bool": True})
    assert len(data) == 6


def test_dataset_set():
    """Test set method of Dataset class"""
    dataset = Dataset()
    test = {
        "MSR[V]": [1, 2, 3],
        "i[V]": [3.0, 4.0, 5.0],
        "q[V]": np.array([3, 4, 5]),
        "phase[deg]": [6.0, 7.0, 8.0],
    }
    dataset.data = test
    assert len(dataset) == 3
    assert (dataset.get_values("MSR", "V") == [1, 2, 3]).all()
    assert (dataset.get_values("i", "V") == [3.0, 4.0, 5.0]).all()
    assert (dataset.get_values("q", "V") == [3, 4, 5]).all()
    assert (dataset.get_values("phase", "deg") == [6.0, 7.0, 8.0]).all()

    dataset1 = Dataset(options=["option1", "option2"])
    test = {"option1": ["one", "two", "three"], "option2": [1, 2, 3]}
    dataset1.data = test
    assert len(dataset1) == 3
    assert (dataset1.get_values("option1") == ["one", "two", "three"]).all()
    assert (dataset1.get_values("option2") == [1, 2, 3]).all()


def test_data_set():
    """Test set method of Data class"""
    data = random_data(5)
    test = {
        "int": [1, 2, 3],
        "float": [3.0, 4.0, 5.0],
        "string": ["one", "two", "three"],
        "bool": [True, False, True],
    }
    data.data = test
    assert len(data) == 3
    assert (data.get_values("int") == [1, 2, 3]).all()
    assert (data.get_values("float") == [3.0, 4.0, 5.0]).all()
    assert (data.get_values("string") == ["one", "two", "three"]).all()
    assert (data.get_values("bool") == [True, False, True]).all()


def test_get_values_dataset():
    """Test get_values method of Dataset class"""
    dataset = random_dataset(5, options=["option"])

    assert (dataset.get_values("option") == dataset.data["option"]).all()
    assert (
        dataset.get_values("MSR", "uV")
        == dataset.data["MSR"].pint.to("uV").pint.magnitude
    ).all()


def test_get_values_data():
    """Test get_values method of Data class"""
    data = random_data(5)
    assert (data.get_values("int") == data.data["int"]).all()
