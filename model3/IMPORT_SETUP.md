# Model 3 module setup

Use Python 3.9 or newer. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install --no-deps .
python model3/training/run_dev392.py --help
python model3/inference/ENGINEER_RUN_PRIMARY_BLIND.py --help
```

Installing the local package makes `model3_core` available to both scripts.
The complete package resides in `model3/training/model3_core`. Its `data.py`
and `model.py` are byte-identical copies of the archived sources retained in
`model3/architecture/model3_core`. No scientific implementation was changed.

These commands validate module loading only. Historical inference still expects
the original private archive layout and original checkpoint hashes; this repair
does not make that workflow compatible with sanitized public checkpoints.
