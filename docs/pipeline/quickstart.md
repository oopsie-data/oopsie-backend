# Quickstart

The pipeline processes the whole dataset in many steps, so it needs a large machine and a token with access to the labs' repos. On LS6, run anything heavier than `--list` on a compute node, not the login node; the [LS6 runbook](../operations/ls6.md) has the Slurm scripts.

```bash
uv sync                                                    # install
export HF_TOKEN=<token>                                    # Hub access for the sync and the hashes

uv run oopsie pipeline run --list                          # targets and their stages
uv run oopsie pipeline run all --explain                   # the nightly run: every stage except the manual ones
uv run oopsie pipeline run qa --explain                    # one target, and whatever it needs that is out of date
uv run oopsie pipeline run stats --set use_hub=false       # override an input; here, work offline

uv run oopsie                                              # every command group and command
uv run oopsie index build --help                           # one command, run on its own
```

`--explain` prints which stages ran and which came from the cache.

## Where results land

The inputs in `configs/pipeline.yaml` default to the LS6 paths:

| What | Where |
|---|---|
| the mirror | `/corral/utexas/IRI26024/oopsiedata` |
| the episode index | `/corral/utexas/IRI26024/episode_index/episode_index.parquet` |
| the pipeline cache | `/corral/utexas/IRI26024/episode_index/pipeline_cache` |
| reports | `/corral/utexas/IRI26024/reports` |
| training sets | `/corral/utexas/IRI26024/training` |

Use `--set data_dir=…`, `--set index_path=…` and so on to work elsewhere; [Inputs](running-targets.md#inputs) lists them all.

## Next

- [Overview](index.md): the mirror, the index, the stages and the rules they follow.
- [Running](running.md): targets or commands, and when to use which.
