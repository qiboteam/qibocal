from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, NewType, Union

from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId
from qibolab.serialize import dump_platform

from qibocal.auto.operation import Data, Results
from qibocal.config import log
from qibocal.protocols.characterization import Operation

Id = NewType("Id", str)
"""Action identifiers type."""
Targets = Union[list[QubitId], list[QubitPairId], list[tuple[QubitId, ...]]]
"""Possible targets for protocol: list of qubits, qubit pairs or nested lists."""
ExperimentId = tuple[Id, int]
"""UID for experiment."""
PLATFORM_DIR = "platform"
"""Directory where platform is dumped."""
DEFAULT_NSHOTS = 100
"""Default number on shots when the platform is not provided."""


@dataclass
class Experiment:
    """Experiment holding a protocol.

    Acts as context for state. Can switch between different state
    to perform corresponding methods.
    """

    id: Id
    """Experiment Id."""
    operation: str | None = None
    """Operation to be performed."""
    iteration: int = 0
    """Task iteration."""
    targets: Targets | None = None
    """Local targets."""
    update: bool = True
    """Local update."""

    # TODO: remove those
    data_time: int = 0
    results_time: int = 0

    parameters: dict[str, Any] | None = None
    """Dicti with protocol parameters."""

    _path: Path = None
    """Path where data and results are stored."""
    _state: State = None
    """Reference to state object."""

    def __post_init__(self) -> None:
        self.transition_to(Pending())

    def transition_to(self, state: State):
        """Change experiment state:"""

        log.info(f"Experiment: Transition to {type(state).__name__}")
        self._state = state
        self._state.experiment = self

    @property
    def raw(self):
        data = {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.name != "_state"
        }
        return data

    @property
    def results(self):
        self.transition_to(Fitted())
        # FIXME: re-introduce pylint
        return self._state.results  # pylint: disable=E1101

    @property
    def data(self):
        self.transition_to(Acquired())
        # FIXME: re-introduce pylint
        return self._state.data  # pylint: disable=E1101

    @property
    def uid(self) -> ExperimentId:
        """Task unique Id."""
        return (self.id, self.iteration)

    @property
    def path(self):
        """Path where experiment is dumped.

        Returns:
            pathlib.Path: path
        """
        return self._path

    @path.setter
    def path(self, new_path: Path):
        self._path = new_path / f"data/{self.id}_{self.iteration}"

    @property
    def protocol(self):
        """Reference to routine object

        Returns:
            Routine: routine object
        """
        return Operation[self.operation].value

    def acquire(self, platform: Platform, targets: Targets) -> None:
        """Protocol acquisition step.

        Args:
            platform (Platform): Qibolab platform
            targets (Targets): Experiment' targets
        """
        if self.targets is None:
            self.targets = targets
        # TODO: check how to handle local and global target
        return self._state.acquire(platform=platform, targets=self.targets)

    def fit(self) -> None:
        """Protocol post-processing step"""
        return self._state.fit()

    def dump(self, path: Path | None = None) -> None:
        """Dump experiment.

        Args:
            path (Optional[Path], optional): Path where experiment if dumped.
            Defaults to None. If None experiment.path is used.

        Returns:
            _type_: _description_
        """
        if not self.path.is_dir():
            self.path.mkdir(parents=True)

        self._state.dump(self.path)

    def update_platform(self, platform: Platform):
        """Performs platform update based on post-processed data.

        Args:
            platform (Platform): Qibolab platform
        """
        self._state.update_platform(platform=platform)


@dataclass
class State:
    """
    Current state of experiment.

    Provide interface for all concrete experiments state.
    """

    _experiment: Experiment = field(init=False)
    """Reference to experiment object."""

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @experiment.setter
    def experiment(self, experiment: Experiment) -> None:
        self._experiment = experiment

    @abstractmethod
    def dump(self, path: Path) -> None:
        """Dump current state of experiment."""

    @abstractmethod
    def acquire(self, platform: Platform, targets: Targets) -> Data:
        """Protocol acquisition step.

        Args:
            platform (Platform): Qibolab platform
            targets (Targets): Experiment' targets

        Returns:
            Data: Acquired Data
        """

    @abstractmethod
    def fit(self) -> Results:
        """Protocol post-processing step

        Returns:
            Results: Post-processed data
        """

    @abstractmethod
    def update_platform(self, platform: Platform) -> None:
        """Performs platform update based on post-processed data.

        Args:
            platform (Platform): Qibolab platform
        """


@dataclass
class Pending(State):
    """State of loaded experiments.

    Hold reference to deserialized parameters.
    Can perform acquisition and fit or update on store data.

    """

    @property
    def parameters(self):
        """Load state from dict"""
        return self.experiment.protocol.parameters_type.load(self.experiment.parameters)

    def acquire(self, platform: Platform, targets: Targets) -> None:
        """Perform acquisition"""
        log.info(f"Performing acquisition for protocol {self.experiment.id}")

        if platform is not None:
            if self.parameters.nshots is None:
                self.experiment.parameters["nshots"] = platform.settings.nshots
            if self.parameters.relaxation_time is None:
                self.experiment.parameters["relaxation_time"] = (
                    platform.settings.relaxation_time
                )
        else:
            if self.parameters.nshots is None:
                self.experiment.parameters["nshots"] = DEFAULT_NSHOTS

        # TODO: fix data_time
        data, self.experiment.data_time = self.experiment.protocol.acquisition(
            self.parameters, platform, targets
        )
        self.experiment.transition_to(Acquired(data))

    def fit(self) -> Results:
        log.info(f"Performing fit {self.experiment.id} on stored data.")
        self.experiment.transition_to(Acquired())

    def update_platform(self, platform) -> None:
        log.info(
            f"Cannot update platform without running fitting on protocol {self.experiment.id}"
        )
        self.experiment.transition_to(Updated(platform))


@dataclass
class Acquired(State):
    """Experiment state after acquiring data."""

    _data: Data | None = None

    @property
    def data(self):
        """Access state's data."""
        if self._data is None:
            Data = self.experiment.protocol.data_type
            self._data = Data.load(self.experiment.path)
        return self._data

    def dump(self, path: Path):
        """Dump data"""
        self.data.save(path)

    def acquire(self) -> None:
        """Perform acquisition"""
        log.info(f"Acquisition for {self.experiment.id} already performed.")

    def fit(self) -> Results:
        log.info(f"Starting fitting on protocol {self.experiment.id}")
        results, self.experiment.results_time = self.experiment.protocol.fit(self.data)
        self.experiment.transition_to(Fitted(results))

    def update_platform(self, platform) -> None:
        log.info(
            f"Performing update on platform for protocol {self.experiment.id} using stored fitted data."
        )
        self.experiment.transition_to(Updated(platform))


@dataclass
class Fitted(State):
    """Experiment state after running fitting."""

    _results: Results | None = None

    @property
    def results(self):
        """Access state's results."""
        if self._results is None:
            Results = self.experiment.protocol.results_type
            self._results = Results.load(self.experiment.path)
        return self._results

    def dump(self, path: Path):
        """Dump results fit."""
        self.results.save(path)

    def acquire(self) -> None:
        """Perform acquisition"""
        log.info(f"Acquisition for {self.experiment.id} already performed.")

    def fit(self) -> None:
        print("Fitting already performed")

    def update_platform(self, platform: Platform) -> None:
        """Platform update."""
        log.info("Updating platform")
        for target in self.experiment.targets:
            try:
                self.experiment.protocol.update(self.results, platform, target)
            except KeyError:
                log.warning(f"Skipping update of qubit {target} due to error in fit.")
        self.experiment.transition_to(Updated(platform))


@dataclass
class Updated(State):
    """Experiment state after platform update."""

    platform: Platform
    """Updated platform"""

    def dump(self, path: Path):
        log.info("Dumping update platform.")
        (path / PLATFORM_DIR).mkdir(parents=True, exist_ok=True)
        dump_platform(self.platform, path / PLATFORM_DIR)
