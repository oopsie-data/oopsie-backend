# Running

There are two ways to run the pipeline's work:

| | [Run a target](running-targets.md) | [Run a command](running-commands.md) |
|---|---|---|
| Command | `oopsie pipeline run <target>` | `oopsie <group> <command>` |
| Runs | a named set of stages, after any stage they depend on that is out of date | one job, with the options you give it |
| Cache | skips stages whose code and inputs are unchanged | none; always runs |
| Inputs | `configs/pipeline.yaml`, overridden with `--set` | the command's own options |
| Use it for | the nightly run and anything that should update the index in order | debugging, one-off runs, helpers that are not stages |

A stage is the pipeline's wrapper around a command: running a target ends up calling the same code as running the commands by hand, in the right order and with the cache. When in doubt, run the target.
