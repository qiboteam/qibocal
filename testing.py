# import pdb

import numpy as np
from qibo import gates, models

from qibocal.calibrations.protocols.experiment import Experiment
from qibocal.calibrations.protocols.test import *
from qibocal.data import Data

# sequence_lengths = [1,2]
# runs = 3
# nshots = 5
# qubit = [0,1]
# measurement_type = "Basis Measurement"
# myshadow = Shadow(len(qubit), measurement_type, sequence_lengths, runs, nshots)

# mygenerator = UIRSNqubitscliffords(len(qubit), invert=False, noisemodel=False)
# myshadow = experimental_protocol(mygenerator, myshadow, nshots=nshots)

# pdb.set_trace()

# from src.qcvv.calibrations.protocol.test import *

# platform =

# standard_rb_test(
#     platform,
#     qubit : list,
#     generator_name,
#     sequence_lengths,
#     runs,
#     nshots,
#     inject_noise
# )


folder = "2022-10-26-002-jadwiga-wilkens"
routine = "dummyrb"
# Load the data into Dataset object.
data_circs = Data.load_data(folder, routine, "pickle", "circuits")
data_probs = Data.load_data(folder, routine, "pickle", "probabilities")
data_samples = Data.load_data(folder, routine, "pickle", "samples")
# Build an Experiment object out of it.
experiment = Experiment.retrieve_from_dataobjects(data_circs, data_samples, data_probs)
