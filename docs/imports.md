# Imported datasets

Imported datasets are bundled with Oopsie-Data but are not lab contributions. `configs/imports.yaml` lists all of them. Every entry is an `external` dataset: it is indexed, QA-checked and available to training recipes, but never public and never counted toward the lab requirements.

An entry takes one of two forms:

```yaml
datasets:
  SOAR:                                   # already converted and on the Hub
    hub: OopsieData-Submissions/SOAR_origin
  my_dataset:                             # converted here
    converter: rlds
    source: /path/to/rlds/1.0.0
    args: [--max-episodes, "1000"]        # optional converter arguments
```

- **`hub`**: a repo of the Hugging Face org that holds a converted dataset. The sync mirrors it like any repo, and the import stage only checks that it is in the mirror.
- **`converter`**: a dataset converted locally into `<data root>/imports/<name>/`, next to the org folder and `annotation/`. It is never uploaded, and the sync never touches it. Its dataset id in the index is `imports/<name>`.

## Converting

```bash
uv run oopsie pipeline run imports                    # through the pipeline
uv run oopsie convert imports --only my_dataset       # directly
sbatch slurm/pipeline.slurm imports                   # on an LS6 compute node
```

A local import is converted once:
1. The converter writes into `.<name>.partial`. That folder is under the pipeline's `stage_dir`, or under `--work-dir` when running directly, and otherwise in the imports folder.
2. When the converter finishes, the folder is moved to `imports/<name>`.

If a run fails, the next run resumes the conversion, because converters skip episodes they have already written. Next to the converted folder, `<name>.inputs` records the converter, source and arguments. If the entry changes later, the run fails instead of converting again. To reconvert, for example after fixing a converter, delete the folder and its record.

The next index build discovers the converted episodes, and QA samples them like every other dataset (`scope.include_external_datasets` in `configs/action_qa_config.yaml`). Because the index and QA run nightly, converting is the only manual step.

## Converters

Converters are listed in `oopsie_backend/conversion/registry.py`, under the names that `converter` uses:

| Name | Module | Source |
|---|---|---|
| `soar` | `convert_soar_data` | SOAR trajectory folders |
| `roboarena` | `convert_roboarena_data` | RoboArena evaluation sessions |
| `ar-aloha` | `convert_ar_aloha_data` | AR-ALOHA recordings |
| `rlds` | `convert_rlds_to_hdf5` | a DROID-style RLDS version folder |

A converter is a command with `main(argv)`. It writes `<episode>.h5` files in the Oopsie-Data format, together with their videos, into `--output-dir`. To add a converter, write the module and add it to the registry with the flag it takes for its source.
