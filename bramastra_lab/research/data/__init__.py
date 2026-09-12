"""Data package: local manifest loading, split validation and group sampling (B03).

The build accepts only operator-supplied local files or explicitly named tiny
fixtures. Missing or empty corpora are ``DATA_NOT_READY`` — never silently
replaced by synthetic data. Downloads are refused.
"""
