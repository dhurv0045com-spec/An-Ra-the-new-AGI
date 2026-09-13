# CS-TRANSFER-001 Colab operator repair

If the first cell prints the exact scientific executable and `CRITICAL BLOBS: PASS 16` but then raises `CalledProcessError` from `tools/validate_cs_transfer_001.py`, do not restart the T4 and do not change Drive state.

The frozen validator must be executed as a module from repository root:

```python
subprocess.run([sys.executable, "-m", "tools.validate_cs_transfer_001"], cwd=REPO, check=True)
```

Then run the five mandatory pytest files from the original first cell. This is an operator import-path correction only; scientific executable remains `a916d1c8d2637abb86d16b1c78e418c95461f3c7`.

See `docs/cymek/cs_transfer_001/OPERATOR_AMENDMENT_2.md` for provenance.
