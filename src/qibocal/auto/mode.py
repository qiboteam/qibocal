from enum import Enum


class ExecutionMode(Enum):
    acquire = "acquire"
    fit = "fit"
    autocalibration = "autocalibration"
    report = "report"
