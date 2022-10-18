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
    assert len(data.df.columns) == 4
    assert list(data.df.columns) == ["MSR", "i", "q", "phase"]

    data1 = Dataset(quantities={"attenuation": "dB"})
    assert len(data1.df.columns) == 5
    assert list(data1.df.columns) == ["attenuation", "MSR", "i", "q", "phase"]

    data2 = Dataset(quantities={"attenuation": "dB"}, options=["option1"])
    assert len(data2.df.columns) == 6
    assert list(data2.df.columns) == [
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
    assert dataset.df.MSR.values.units == "volt"

    dataset1 = Dataset(quantities={"frequency": "Hz"})
    assert dataset1.df.frequency.values.units == "hertz"

    with pytest.raises(UndefinedUnitError):
        dataset2 = Dataset(quantities={"fake_unit": "fake"})


def test_dataset_add():
    """Test add method of Dataset"""
    dataset = random_dataset(5)
    assert len(dataset) == 5

    dataset1 = Dataset(quantities={"attenuation": "dB"})
    msr, i, q, phase, att = np.random.rand(len(dataset1.df.columns))
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
    msr, i, q, phase = np.random.rand(len(dataset2.df.columns))
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


def test_get_values_dataset():
    """Test get_values method of Dataset class"""
    dataset = random_dataset(5, options=["option"])

    assert (dataset.get_values("option") == dataset.df["option"]).all()
    assert (
        dataset.get_values("MSR", "uV")
        == dataset.df["MSR"].pint.to("uV").pint.magnitude
    ).all()


def test_get_values_data():
    """Test get_values method of Data class"""
    data = random_data(5)
    assert (data.get_values("int") == data.df["int"]).all()
