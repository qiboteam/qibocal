import logging
from pathlib import Path

from qibocal.fitting.classifier import plots, run

logging.basicConfig(level=logging.INFO)
data_path = Path("calibrate_qubit_states/data.csv")
base_dir = Path("results")
base_dir.mkdir(exist_ok=True)

for qubit in range(1, 5):
    print(f"QUBIT: {qubit}")
    qubit_dir = base_dir / f"qubit{qubit}"
    table, y_test, x_test = run.train_qubit(
        data_path, base_dir, qubit, classifiers=["qblox_fit"]
    )
    run.dump_benchmarks_table(table, qubit_dir)

    plots.plot_table(table, qubit_dir)
