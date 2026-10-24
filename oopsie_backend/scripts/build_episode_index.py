"""Alias for ``oopsie_backend.index.build``."""

import runpy

if __name__ == "__main__":
    runpy.run_module("oopsie_backend.index.build", run_name="__main__", alter_sys=True)
