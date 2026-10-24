# Annotator guide

You have your own folder in the private Hugging Face dataset `OopsieData-Submissions/oopsie-annotation`. It holds two batches of robot episodes, `batch_1` and `batch_2`. Annotate every episode in both batches, then upload your annotated files.

## Setup

1. Send your Hugging Face username to the coordinator, so you are added to the `OopsieData-Submissions` organization.
2. Install the Hugging Face CLI with `pip install -U huggingface_hub`. This provides the `hf` command.
3. Install the `oopsie-data` annotation tool, as described in its README.
4. Log in with `hf auth login`.
5. Open the repo's `README.md` on Hugging Face and find your name in the table. The **Folder** column gives your folder name, e.g. `jiahui_karen_chen`. Replace `<folder>` below with it.

## Download

```bash
hf download OopsieData-Submissions/oopsie-annotation --repo-type dataset \
  --include "<folder>/*" --local-dir oopsie-annotation
```

This downloads only your folder, which takes a few GB. If the download is interrupted, run the same command again and it resumes.

## Annotate

Annotate both batches, using your full name exactly as it appears in the README table:

```bash
oopsie-data annotate --samples-dir oopsie-annotation/<folder>/batch_1/episodes --annotator-name "<your name>"
oopsie-data annotate --samples-dir oopsie-annotation/<folder>/batch_2/episodes --annotator-name "<your name>"
```

Episodes are numbered in the order they should be annotated. You can stop and continue later, because annotations are saved into the `.h5` files.

## Upload

```bash
hf upload OopsieData-Submissions/oopsie-annotation oopsie-annotation/<folder> <folder> \
  --repo-type dataset --include "*.h5" --create-pr
```

This opens a pull request with your annotated files. Uploading partial progress is fine, and so is uploading again later: each upload opens a new pull request, and the most recent annotation of each episode counts.

## Rules

- Upload only your own folder, and only the `.h5` files, as in the command above. A pull request that changes anything else is rejected.
- Do not rename, move or delete files.
- Do not open the files with any tool other than `oopsie-data annotate`.
- If something fails or looks wrong, contact the coordinator.
