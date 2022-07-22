# -*- coding: utf-8 -*-
import numpy as np
import yaml

from qcvv.config import raise_error


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
            raise_error(RuntimeError, "Error when adding data.")

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

    def to_dict(self):
        return {"name": self.name, "unit": self.unit, "data": self.data}


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
                raise_error(RuntimeError, f"Format of {quantities} is not valid.")
        self.points = points

    def add(self, msr, i, q, phase, quantities=None):
        self.container["MSR"].add("MSR", "V", float(msr))
        self.container["i"].add("i", "V", float(i))
        self.container["q"].add("q", "V", float(q))
        self.container["phase"].add("phase", "deg", float(phase))

        if quantities is not None:
            if isinstance(quantities, tuple):
                self.container[quantities[0]].add(
                    quantities[0], quantities[1], float(quantities[2])
                )
            elif isinstance(quantities, list):
                for item in quantities:
                    self.container[item[0]].add(item[0], item[1], float(item[2]))
            else:
                raise_error(RuntimeError, f"Format of {quantities} is not valid.")

    def compute_software_average(self, quantities):
        avg_dataset = self.__class__()
        if isinstance(quantities, str):
            avg_dataset.add(
                self.container[quantities].name,
                self.container[quantities].unit,
                np.unique(self.container[quantities].data),
            )
            total = len(avg_dataset.container[quantities].data)
        elif isinstance(quantities, list):
            total = 1
            for quantity in quantities:
                avg_dataset.add(
                    self.container[quantity].name,
                    self.container[quantity].unit,
                    np.unique(self.container[quantity].data),
                )
                total *= len(avg_dataset.container[quantity].data)
        else:
            raise_error(RuntimeError, f"Format of {quantities} is not valid.")

        for j in range(total):
            for name, unit in [("MSR", "V"), ("i", "V"), ("q", "V"), ("phase", "deg")]:
                avg_dataset.container[name].add(
                    name,
                    unit,
                    float(
                        np.mean(
                            [
                                self.container[name].data[i]
                                for i in range(total)
                                if i % 5 == j
                            ]
                        )
                    ),
                )

        return avg_dataset

    def to_yaml(self, path, name="data"):
        output = {}
        for i, c in self.container.items():
            output[i] = c.to_dict()
        with open(f"{path}/{name}.yml", "w") as f:
            yaml.dump(output, f)
