"""Alias for ``oopsie_backend.stats.render_internal``."""

import runpy

if __name__ == "__main__":
    runpy.run_module("oopsie_backend.stats.render_internal", run_name="__main__", alter_sys=True)
