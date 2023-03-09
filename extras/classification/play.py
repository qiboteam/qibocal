from pathlib import Path

from qibocal.fitting.classifier import (
    plots,
    run,
)

data_path = Path("calibrate_qubit_states/data.csv")
base_dir = Path("_results2")
try:
    base_dir.mkdir()
except:
    pass
# qubit = 1
for qubit in range(1, 5):
    qubit_dir = base_dir / f"qubit{qubit}"
    classifiers = [linear_svm]
    table, y_test, x_test = run.train_qubit(data_path, base_dir, qubit)
    run.dump_benchmarks_table(table, qubit_dir)
    plots.plot_table(table, qubit_dir)
    plots.plot_conf_matr(y_test, qubit_dir)
    # plots.plot_roc_curves(x_test, y_test, qubit_dir)
    # plot.plot_models_results( x_train, x_test, y_test, base_dir, classifiers)
