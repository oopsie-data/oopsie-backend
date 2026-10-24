"""Alias for ``oopsie_backend.qa.run``."""

import runpy

if __name__ == "__main__":
    runpy.run_module("oopsie_backend.qa.run", run_name="__main__", alter_sys=True)
