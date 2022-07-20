# -*- coding: utf-8 -*-
import yaml


class MeasuredQuantity:
    """Data Structure for collecting all the information for a quantity
    measured during a calibration routine"""

    def __init__(self, name=None, unit=None, data=None):
        self.unit = unit
        self.name = name
        self.data = []
        if data is not None:
            self.data.append(data)

    def add(self, *data):
        if isinstance(data, MeasuredQuantity):
            assert self.unit != data.unit, "Unit of measure are not compatible"
            assert self.name != data.name, "Impossible to compare different quantities"
            self.data += data.data
        elif len(data) == 3:
            assert self.unit != data[0], "Unit of measure are not compatible"
            assert self.name != data[1], "Impossible to compare different quantities"
            self.data.append(data[2])
        else:
            raise RuntimeError("Error when adding data.")

    def __repr__(self):
        return f"{self.name} ({self.unit}): {self.data}"

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


class Dataset:
    """First prototype of dataset for calibration routines."""

    def __init__(self, points=100, quantities=None):
        self.container = {}
        self.container["MSR"] = MeasuredQuantity(name="MSR", unit="V")
        self.container["i"] = MeasuredQuantity(name="i", unit="V")
        self.container["q"] = MeasuredQuantity(name="q", unit="V")
        self.container["phase"] = MeasuredQuantity(name="phase", unit="deg")
        if quantities is not None:
            if isinstance(quantities, tuple):
                self.container[quantities[0]] = MeasuredQuantity(*quantities)
            elif isinstance(quantities, list):
                for item in quantities:
                    self.container[item[0]] = MeasuredQuantity(*item)
            else:
                raise RuntimeError(f"Format of {quantities} is not valid.")
        self.points = points

    def add(self, results, qubit, pulse_serial, quantities=None):
        msr, i, q, phase = results[qubit][pulse_serial]
        self.container["MSR"].add("MSR", "V", float(msr))
        self.container["i"].add("i", "V", float(i))
        self.container["q"].add("q", "V", float(q))
        self.container["phase"].add("phase", "deg", float(phase))

        if quantities is not None:
            if isinstance(quantities, tuple):
                self.container[quantities[0]].add(
                    quantities[0], quantities[1], quantities[2]
                )
            elif isinstance(quantities, list):
                for item in quantities:
                    self.container[item[0]].add(item[0], item[1], item[2])
            else:
                raise RuntimeError(f"Format of {quantities} is not valid.")

    def save(self, file):
        to_yaml = {}
        for i in self.container:
            to_yaml[i] = {}
            to_yaml[i]["name"] = self.container[i].name
            to_yaml[i]["unit"] = self.container[i].unit
            to_yaml[i]["data"] = self.container[i].data
        with open(file, "w") as f:
            yaml.dump(self._prepare_yaml(), f)
