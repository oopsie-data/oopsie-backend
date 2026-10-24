"""Alias for ``oopsie_backend.annotation.collect_returns``."""

import runpy

if __name__ == "__main__":
    runpy.run_module("oopsie_backend.annotation.collect_returns", run_name="__main__", alter_sys=True)
