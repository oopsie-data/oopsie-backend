"""Alias for ``oopsie_backend.splits.sample_train``."""

import runpy

if __name__ == "__main__":
    runpy.run_module("oopsie_backend.splits.sample_train", run_name="__main__", alter_sys=True)
