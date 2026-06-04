Inspect one or more oopsiedata HDF5 files and print a coloured tree to the terminal.

Run:
```bash
python -m oopsie_backend.inspect_hdf5 $ARGUMENTS
```

`$ARGUMENTS` is passed directly — it can be a single path, a glob, or multiple paths. The script prints, for each file:
- Root attributes (schema, episode_id, language_instruction, …)
- Full dataset tree with shapes `(T × D)  dtype` for arrays and inline values for string datasets
- Group attributes (episode_annotations metadata, etc.) prefixed with `@`
