"""Alias for ``oopsie_backend.ingest.sync``."""

import runpy

if __name__ == "__main__":
    runpy.run_module("oopsie_backend.ingest.sync", run_name="__main__", alter_sys=True)
