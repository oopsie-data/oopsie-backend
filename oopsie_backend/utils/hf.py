"""Shared Hugging Face Hub helpers for oopsie-backend scripts."""

from __future__ import annotations

from urllib.parse import urlparse

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder


def parse_repo_arg(repo: str, repo_type: str) -> tuple[str, str]:
    """Return ``(repo_id, repo_type)`` with ``repo_type`` in ``{"dataset","model"}``."""
    s = repo.strip().rstrip("/")

    if s.startswith("http"):
        u = urlparse(s)
        host = u.netloc.lower()
        if "huggingface.co" not in host and "hf.co" not in host:
            raise ValueError(f"Unsupported URL host: {u.netloc!r}")
        parts = [p for p in u.path.strip("/").split("/") if p]
        if parts[:1] == ["datasets"]:
            if len(parts) < 3:
                raise ValueError(
                    "Dataset URL must look like "
                    "https://huggingface.co/datasets/<namespace>/<repo-name>"
                )
            return "/".join(parts[1:3]), "dataset"
        if len(parts) == 1:
            raise ValueError(
                f"{s!r} points at a single path segment ({parts[0]!r}). "
                "That is usually a Hugging Face *organization* page, not a dataset. "
                "Open your dataset in the browser and copy the id from the URL "
                "`https://huggingface.co/datasets/<namespace>/<repo-name>`, "
                "or pass `--repo <namespace>/<repo-name>`."
            )
        if len(parts) == 2:
            if repo_type != "model":
                raise ValueError(
                    f"{s!r} has no `/datasets/` segment. For a dataset, use the full URL "
                    "`https://huggingface.co/datasets/<namespace>/<repo-name>` "
                    "or `--repo <namespace>/<repo-name>`. "
                    "If this file really lives in a *model* repo, pass `--repo-type model`."
                )
            return "/".join(parts), "model"
        raise ValueError(f"Could not parse repository id from URL: {s}")

    if "/" not in s:
        raise ValueError(
            f"Hugging Face repo ids look like `namespace/repo`, not {s!r}. "
            f"The page https://huggingface.co/{s} is typically an organization, "
            "not where episode files are stored. Use your dataset's "
            "`namespace/repo` from `https://huggingface.co/datasets/...`."
        )
    return s, repo_type


def parse_org_arg(org: str) -> str:
    """Return the Hugging Face *organization* slug used with ``list_datasets``."""
    s = org.strip().rstrip("/")
    if s.startswith("http"):
        u = urlparse(s)
        if "huggingface.co" not in u.netloc.lower() and "hf.co" not in u.netloc.lower():
            raise ValueError(f"Unsupported URL host: {u.netloc!r}")
        parts = [p for p in u.path.strip("/").split("/") if p]
        if not parts:
            raise ValueError(f"Empty path in organization URL: {org!r}")
        if parts[0] == "datasets":
            raise ValueError(
                "That URL is a dataset page. Use `--repo` for one dataset, "
                "or `--org` with your organization slug (e.g. OopsieData-Submissions)."
            )
        if parts[0] == "organizations" and len(parts) >= 2:
            return parts[1]
        if len(parts) == 1:
            return parts[0]
        raise ValueError(
            f"Could not parse organization from {org!r}. "
            "Use a single-segment profile URL like https://huggingface.co/MyOrg "
            "or pass `--org MyOrg`."
        )
    return s


def split_repo_id(repo_id: str) -> tuple[str, str]:
    """Return ``(namespace, repo_name)`` from ``namespace/repo_name``."""
    namespace, repo_name = repo_id.split("/", 1)
    return namespace, repo_name


def list_dataset_repos_for_org(author: str, token: str | None) -> list[str]:
    """Return dataset repo ids ``namespace/name`` owned by ``author``."""
    api = HfApi(token=token)
    out: list[str] = []
    for info in api.list_datasets(author=author, token=token):
        rid = getattr(info, "id", None) or getattr(info, "dataset_id", None)
        if rid:
            out.append(str(rid))
    return sorted(set(out))


def list_dataset_created_dates_for_org(author: str, token: str | None) -> dict[str, str | None]:
    """Return ``{repo_id: created_at}`` (ISO 8601 UTC, or ``None``) for every dataset
    repo owned by ``author``, straight from the Hub's repo metadata."""
    api = HfApi(token=token)
    out: dict[str, str | None] = {}
    for info in api.list_datasets(author=author, token=token):
        rid = getattr(info, "id", None) or getattr(info, "dataset_id", None)
        if not rid:
            continue
        created = getattr(info, "created_at", None)
        out[str(rid)] = created.isoformat().replace("+00:00", "Z") if created else None
    return out


def list_repo_files(repo: str, token: str | None, repo_type: str) -> list[str]:
    """Return every repo-relative file path."""
    api = HfApi(token=token)

    def walk(path: str | None):
        for entry in api.list_repo_tree(
            repo_id=repo,
            path_in_repo=path,
            repo_type=repo_type,
            recursive=False,
            expand=False,
        ):
            if isinstance(entry, RepoFolder):
                yield from walk(entry.path)
            elif isinstance(entry, RepoFile):
                yield entry.path

    return sorted(walk(None))


def list_h5_files(repo: str, token: str | None, repo_type: str) -> list[str]:
    """Return every ``.h5`` repo-relative path."""
    return [p for p in list_repo_files(repo, token, repo_type) if p.lower().endswith(".h5")]
