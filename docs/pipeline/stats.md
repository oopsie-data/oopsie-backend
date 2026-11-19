# Stats

The stats stage computes the numbers behind the public dashboard on the Oopsie-Data website and the internal report the team uses to follow each lab's progress.

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `stats` | `stats` | `critical_episodes`, `stats_config` | the website's `stats.json`; `public_stats.json`, `.html`, `internal_stats.json`, `.html` in `reports_dir` |

## What is counted

The stats count readable, on-disk episodes that are neither a duplicate copy nor critical. They are computed from the index alone, so they need no Hub token and read no h5 files.

- **Public stats** cover contributor repos with a real lab id. Placeholder lab ids, listed in `configs/stats_config.yaml` under `labs.public_excluded`, are left out. The nightly job commits the resulting `stats.json` to the `oopsie-docs` site, where the dashboard reads it.
- **Internal stats** cover every repo and add what the team needs: an inventory of all dataset repos with their creation dates, episode counts and hours, critical episodes per lab with their reasons, and episodes credited to placeholder operator names.

The internal report lists repos that have no episodes yet as well, for labs that registered but have not submitted. Their creation dates come from the Hub when a token is available; otherwise each dataset's earliest episode timestamp is used.

## Lab requirements

Each lab must contribute enough annotated episodes. A lab is `alpha` if any of its repos was created before `requirements.alpha_cutoff_date`, and `beta` otherwise. A lab passes its tier when its episodes, summed over all its repos and operators, meet every enabled clause of that tier in `configs/stats_config.yaml`. The internal inventory shows the result as a colored dot per lab. Changes to the thresholds take effect on the next run.

## Running it by hand

```bash
uv run oopsie stats compute-local --index episode_index.parquet --data-dir /corral/utexas/IRI26024/oopsiedata
```

| Option | Effect |
|---|---|
| `--no-public-report` | skip `public_stats.html` |
| `--no-internal-output` | skip the internal JSON and HTML |
| `--no-internal-report` | skip `internal_stats.html` only |

`oopsie stats compute-hf --repo <repo>` computes the same stats by streaming a repo's h5 files from the Hub, without a mirror.
