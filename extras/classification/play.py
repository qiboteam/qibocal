from qibocal.fitting.classifier import run 
from pathlib import Path

data_path = Path('calibrate_qubit_states/data.csv')
base_dir = Path('results')
qubit = 1
table = run.train_qubit(data_path, base_dir, qubit)
run.dump_benchmarks_table(table,base_dir / f"qubit{qubit}")
