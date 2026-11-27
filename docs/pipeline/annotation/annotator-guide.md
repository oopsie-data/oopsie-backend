# Annotator guide

You have your own folder in the private Hugging Face dataset `OopsieData-Submissions/oopsie-annotation`. It holds one or more batches of robot episodes, `batch_1`, `batch_2`, and so on. Annotate every episode in each batch, then upload your annotated files with `oopsie-data annotation upload`. When you have finished and want to annotate more, ask the coordinator for another batch; it appears as the next `batch_<n>` in your folder.

## Setup

1. Send your Hugging Face username to the coordinator, so you are given access to the annotation repos.
2. Install the `oopsie-data` tool, as described in its README.
3. Create a Hugging Face token with read access at https://huggingface.co/settings/tokens. The tool asks for it and does not store it.

## Download

```bash
oopsie-data annotation download
```

The tool asks for your full name and your token, then downloads only your folder, which takes a few GB, into `oopsie-annotation/`. Type your name as you gave it to the coordinator; it is also listed in the repo's `README.md`. If the download is interrupted, run the same command again and it resumes.

## Annotate

At the end of the download, the tool prints one command per batch. Each has the form

```bash
oopsie-data annotate --samples-dir oopsie-annotation/<folder>/batch_1/episodes --annotator-name "<your name>"
```

Run each of them, and keep the annotator name exactly as printed, because the upload only sends episodes annotated under that name. Episodes are numbered in the order they should be annotated. You can stop and continue later, because annotations are saved into the `.h5` files.

## Upload

```bash
oopsie-data annotation upload
```

The tool shows how many episodes of each batch you annotated, asks for your token, and opens a pull request with your annotated files on `OopsieData-Submissions/oopsie-annotation-returns`. Uploading partial progress is fine, and so is uploading again later: each upload opens a new pull request, and the most recent annotation of each episode counts.

## Rules

- Upload only with `oopsie-data annotation upload`. A pull request that changes anything other than your own `.h5` files is rejected.
- Do not rename, move or delete files.
- Do not open the files with any tool other than `oopsie-data annotate`.
- If something fails or looks wrong, contact the coordinator.
