# Code checks

The Python suite uses synthetic records and a Python fixture process. It does
not launch MA-CETSP, Gurobi or model fitting. From the repository root:

```console
python -c "from pathlib import Path; Path('build-task2-validation').mkdir(exist_ok=True)"
python -m unittest discover -s tests -p "test_*.py" -v
```

The four CMake unit-test targets cover pruning rules, candidate/lifetime ledgers,
native Cox scoring and its lifecycle. Build these targets separately from the
solver and use CTest. Do not run an experiment campaign to verify saved tables;
use the saved-result command in the main README.

The editable `tools/experiment_control.py` includes a bounded retry for transient
Windows report-file locks discovered during publication checks. This changes
report I/O only. The deployed frozen snapshots and completed records in
`reproducibility/revision_20261109/` remain byte-identical to the source bundle;
the original neural campaign's I/O maintenance adapter is included there.
