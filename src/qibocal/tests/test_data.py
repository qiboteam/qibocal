"""Some tests for the Data and DataUnits class"""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from pint import DimensionalityError, UndefinedUnitError

from qibocal.data import AbstractData, Data, DataUnits


def random_data_units(length, options=None):
    data = DataUnits(options=options)
    for l in range(length):
        msr, i, q, phase = np.random.rand(4)
        pulse_sequence_result = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
        }
        add_options = {}
        if options is not None:
            for option in options:
                add_options[option] = l
        data.add({**pulse_sequence_result, **add_options})

    return data


def data_units_dummy(length, options=None):
    data = DataUnits(options=options)
    for l in range(length):
        pulse_sequence_result = {
            "MSR[V]": float(l),
            "i[V]": float(l),
            "q[V]": float(l),
            "phase[rad]": float(l),
        }
        add_options = {}
        if options is not None:
            for option in options:
                add_options[option] = float(l)
        data.add({**pulse_sequence_result, **add_options})

    return data


def random_data(length):
    data = Data()
    for i in range(length):
        data.add(
            {
                "int": int(i),
                "float": float(i),
                "string": str(f"hello{i}"),
                "bool": bool(i),
            }
        )
    return data


def data_dummy(length):
    data = Data()
    for i in range(length):
        data.add(
            {
                "num1": 0.0,
                "num2": 1.0,
            }
        )
    return data


def test_data_units_initialization():
    """Test DataUnits constructor"""
    data = DataUnits(name="data")
    assert data.name == "data"
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


def test_data_initialization():
    """Test initialization of class Data"""
    quantities_test = ["test"]
    data = Data(quantities=quantities_test)
    data.add(
        {
            "test": 0,
        }
    )
    assert data.quantities == quantities_test


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
            "phase[rad]": phase,
            "attenuation[dB]": att,
        }
    )
    assert len(data_units1) == 1

    data_units1.add(
        {
            "MSR[V]": 0,
            "i[V]": 0.0,
            "q[V]": 0.0,
            "phase[rad]": 0,
            "attenuation[dB]": 1,
        }
    )
    assert len(data_units1) == 2

    data_units2 = DataUnits()
    msr, i, q, phase = np.random.rand(len(data_units2.df.columns))
    with pytest.raises(DimensionalityError):
        data_units2.add({"MSR[dB]": msr, "i[V]": i, "q[V]": q, "phase[rad]": phase})

    with pytest.raises(UndefinedUnitError):
        data_units2.add({"MSR[test]": msr, "i[V]": i, "q[V]": q, "phase[rad]": phase})

    data_units3 = random_data_units(10, options=["test"])
    assert len(data_units3) == 10


def test_data_add():
    """Test add method of Data class"""
    data = random_data(5)
    assert len(data) == 5
    data.add({"int": 123, "float": 123.456, "string": "123", "bool": True})
    assert len(data) == 6


def test_data__add__():
    data0 = Data()
    data0.add(
        {
            "col1": 0,
            "col2": 0,
        }
    )
    data1 = Data()
    data1.add(
        {
            "col1": 1,
            "col2": 1,
        }
    )
    data0.__add__(data1)
    data_results = Data()
    for i in [0, 1]:
        data_results.add(
            {
                "col1": int(i),
                "col2": int(i),
            }
        )
    assert data0.df.equals(data_results.df)


def test_data_units_load_data_from_dict():
    """Test set method of DataUnits class"""
    data_units = DataUnits()
    test = {
        "MSR[V]": [1, 2, 3],
        "i[V]": [3.0, 4.0, 5.0],
        "q[V]": np.array([3, 4, 5]),
        "phase[rad]": [6.0, 7.0, 8.0],
    }
    data_units.load_data_from_dict(test)
    assert len(data_units) == 3
    assert (data_units.get_values("MSR", "V") == [1, 2, 3]).all()
    assert (data_units.get_values("i", "V") == [3.0, 4.0, 5.0]).all()
    assert (data_units.get_values("q", "V") == [3, 4, 5]).all()
    assert (data_units.get_values("phase", "rad") == [6.0, 7.0, 8.0]).all()

    data_units1 = DataUnits(options=["option1", "option2"])
    test = {"option1": ["one", "two", "three"], "option2": [1, 2, 3]}
    data_units1.load_data_from_dict(test)
    assert len(data_units1) == 3
    assert (data_units1.get_values("option1") == ["one", "two", "three"]).all()
    assert (data_units1.get_values("option2") == [1, 2, 3]).all()


def test_df_data_units():
    """Test the method df in DataUnit class"""
    data = DataUnits()
    with pytest.raises(TypeError):
        data.df = 0


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


def test_save_open_data_units_csv():
    """Test to_csv and load_data methods of DataUnits"""
    path = "test_folder/test_subfolder/test_routine"
    if not os.path.isdir(path):
        os.makedirs(path)
    data_units = data_units_dummy(5)
    data_units.to_csv(path)
    isExist = os.path.exists(f"{path}/{data_units.name}.csv")
    assert isExist is True
    with pytest.raises(ValueError):
        data_upload = DataUnits().load_data(
            "test_folder", "test_subfolder", "test_routine", "txt", "data"
        )
    data_upload = DataUnits().load_data(
        "test_folder", "test_subfolder", "test_routine", "csv", "data"
    )
    shutil.rmtree("test_folder")
    pd.testing.assert_frame_equal(data_upload.df, data_units.df)


def test_save_open_data_units_pickle():
    """Test to_csv and load_data methods of DataUnits"""
    folder = "test_folder/data/test_routine"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    data_units = random_data_units(5)
    data_units.to_pickle(folder)
    data_upload = DataUnits().load_data(
        "test_folder", "data", "test_routine", "pickle", "data"
    )
    shutil.rmtree("test_folder")
    assert data_units.df.equals(data_upload.df)


def test_save_open_data_csv():
    """Test to_csv and load_data methods of Data"""
    path = "test_folder/data/test_routine"
    if not os.path.isdir(path):
        os.makedirs(path)
    data = data_dummy(5)
    data.to_csv(path)
    isExist = os.path.exists(f"{path}/{data.name}.csv")
    assert isExist is True
    with pytest.raises(ValueError):
        data_upload = Data().load_data(
            "test_folder", "data", "test_routine", "txt", "data"
        )
    data_upload = Data().load_data("test_folder", "data", "test_routine", "csv", "data")
    assert data.df.equals(data_upload.df)


def test_save_open_data_pickle():
    """Test to_pickle and load_data methods of Data"""
    folder = "test_folder/data/test_routine"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    data = random_data(5)
    data.to_pickle(folder)
    data_upload = Data().load_data(
        "test_folder", "data", "test_routine", "pickle", "data"
    )
    shutil.rmtree("test_folder")
    assert data.df.equals(data_upload.df)


def test_load_data_from_dict_data_units():
    """Test load_data_from_dict method of DataUnits"""
    data_units = data_units_dummy(5)

    test_dict = {key: [0, 1, 2, 3] for key in data_units.df.columns}
    data_units.load_data_from_dict(test_dict)

    for column in data_units.df.columns:  # pylint: disable=E1101
        assert (data_units.get_values(column).to_numpy() == [0, 1, 2, 3]).all()


def test_load_data_from_dict_data():
    """Test load_data_from_dict method of Data"""
    data = data_dummy(5)
    test_dict = {key: [0, 1, 2, 3] for key in data.df.columns}
    data.load_data_from_dict(test_dict)

    for column in data.df.columns:  # pylint: disable=E1101
        assert (data.get_values(column) == [0, 1, 2, 3]).all()
