"""Some tests for the Data and DataUnits class"""
import numpy as np
import pytest
from pint import DimensionalityError, UndefinedUnitError

from qibocal.data import Data, DataUnits


def random_data_units(length, options=None):
    data = DataUnits(options=options)
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
    """Test DataUnits constructor"""
    data = DataUnits()
    assert len(data.df.columns) == 4
    assert list(data.df.columns) == [  # pylint: disable=E1101
        "MSR",
        "i",
        "q",
        "phase",
    ]

    data1 = DataUnits(quantities={"attenuation": "dB"})
    assert len(data1.df.columns) == 5
    assert list(data1.df.columns) == [  # pylint: disable=E1101
        "attenuation",
        "MSR",
        "i",
        "q",
        "phase",
    ]

    data2 = DataUnits(quantities={"attenuation": "dB"}, options=["option1"])
    assert len(data2.df.columns) == 6
    assert list(data2.df.columns) == [  # pylint: disable=E1101
        "option1",
        "attenuation",
        "MSR",
        "i",
        "q",
        "phase",
    ]


def test_data_units_units():
    """Test units of measure in DataUnits"""
    data_units = DataUnits()
    assert data_units.df.MSR.values.units == "volt"

    data_units1 = DataUnits(quantities={"frequency": "Hz"})
    assert data_units1.df.frequency.values.units == "hertz"

    with pytest.raises(UndefinedUnitError):
        data_units2 = DataUnits(quantities={"fake_unit": "fake"})


def test_data_units_add():
    """Test add method of DataUnits"""
    data_units = random_data_units(5)
    assert len(data_units) == 5

    data_units1 = DataUnits(quantities={"attenuation": "dB"})
    msr, i, q, phase, att = np.random.rand(len(data_units1.df.columns))
    data_units1.add(
        {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[deg]": phase,
            "attenuation[dB]": att,
        }
    )
    assert len(data_units1) == 1

    data_units1.add(
        {
            "MSR[V]": 0,
            "i[V]": 0.0,
            "q[V]": 0.0,
            "phase[deg]": 0,
            "attenuation[dB]": 1,
        }
    )
    assert len(data_units1) == 2

    data_units2 = DataUnits()
    msr, i, q, phase = np.random.rand(len(data_units2.df.columns))
    with pytest.raises(DimensionalityError):
        data_units2.add({"MSR[dB]": msr, "i[V]": i, "q[V]": q, "phase[deg]": phase})

    with pytest.raises(UndefinedUnitError):
        data_units2.add({"MSR[test]": msr, "i[V]": i, "q[V]": q, "phase[deg]": phase})

    data_units3 = random_data_units(10, options=["test"])
    assert len(data_units3) == 10


def test_data_add():
    """Test add method of Data class"""
    data = random_data(5)
    assert len(data) == 5
    data.add({"int": 123, "float": 123.456, "string": "123", "bool": True})
    assert len(data) == 6


def test_data_units_load_data_from_dict():
    """Test set method of DataUnits class"""
    data_units = DataUnits()
    test = {
        "MSR[V]": [1, 2, 3],
        "i[V]": [3.0, 4.0, 5.0],
        "q[V]": np.array([3, 4, 5]),
        "phase[deg]": [6.0, 7.0, 8.0],
    }
    data_units.load_data_from_dict(test)
    assert len(data_units) == 3
    assert (data_units.get_values("MSR", "V") == [1, 2, 3]).all()
    assert (data_units.get_values("i", "V") == [3.0, 4.0, 5.0]).all()
    assert (data_units.get_values("q", "V") == [3, 4, 5]).all()
    assert (data_units.get_values("phase", "deg") == [6.0, 7.0, 8.0]).all()

    data_units1 = DataUnits(options=["option1", "option2"])
    test = {"option1": ["one", "two", "three"], "option2": [1, 2, 3]}
    data_units1.load_data_from_dict(test)
    assert len(data_units1) == 3
    assert (data_units1.get_values("option1") == ["one", "two", "three"]).all()
    assert (data_units1.get_values("option2") == [1, 2, 3]).all()


def test_data_load_data_from_dict():
    """Test set method of Data class"""
    data = random_data(5)
    test = {
        "int": [1, 2, 3],
        "float": [3.0, 4.0, 5.0],
        "string": ["one", "two", "three"],
        "bool": [True, False, True],
    }
    data.load_data_from_dict(test)
    assert len(data) == 3
    assert (data.get_values("int") == [1, 2, 3]).all()
    assert (data.get_values("float") == [3.0, 4.0, 5.0]).all()
    assert (data.get_values("string") == ["one", "two", "three"]).all()
    assert (data.get_values("bool") == [True, False, True]).all()


def test_get_values_data_units():
    """Test get_values method of DataUnits class"""
    data_units = random_data_units(5, options=["option"])

    assert (data_units.get_values("option") == data_units.df["option"]).all()
    assert (
        data_units.get_values("MSR", "uV")
        == data_units.df["MSR"].pint.to("uV").pint.magnitude
    ).all()


def test_get_values_data():
    """Test get_values method of Data class"""
    data = random_data(5)
    assert (data.get_values("int") == data.df["int"]).all()
