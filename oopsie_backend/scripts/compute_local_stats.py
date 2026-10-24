"""Alias for ``oopsie_backend.stats.compute_local``."""

import runpy

if __name__ == "__main__":
    runpy.run_module("oopsie_backend.stats.compute_local", run_name="__main__", alter_sys=True)
