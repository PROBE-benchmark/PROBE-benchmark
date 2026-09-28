from __future__ import annotations

import csv
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def _yaml_scalar(value: str) -> Any:
    """Parse the small, dependency-free YAML subset used by configs.

    Model files intentionally use ordinary YAML mappings and scalar values.
    Supporting that subset in the standard library keeps a fresh clone
    runnable without requiring a package manager or hiding the configuration
    contract behind an optional dependency.
    """

    value = value.strip()
    if not value:
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parts = [part.strip() for part in value[1:-1].split(",") if part.strip()]
            return [_yaml_scalar(part) for part in parts]
    if value.startswith(("'", '"')) and value.endswith(value[0]):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _strip_yaml_comment(line: str) -> str:
    quoted: str | None = None
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            if quoted == char:
                quoted = None
            elif quoted is None:
                quoted = char
        elif char == "#" and quoted is None and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line.rstrip()


def _simple_yaml(text: str) -> dict[str, Any]:
    entries = []
    for raw in text.splitlines():
        line = _strip_yaml_comment(raw).expandtabs(2)
        if not line.strip() or line.lstrip().startswith("---"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        entries.append((indent, line.strip()))
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for number, (indent, content) in enumerate(entries, 1):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML line {number}: list item has no list parent")
            parent.append(_yaml_scalar(content[2:]))
            continue
        key, separator, raw_value = content.partition(":")
        if not separator or not key.strip():
            raise ValueError(f"YAML line {number}: expected key: value")
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            if not isinstance(parent, dict):
                raise ValueError(f"YAML line {number}: mapping has no mapping parent")
            parent[key] = _yaml_scalar(raw_value)
            continue
        next_indent = entries[number][0] if number < len(entries) else -1
        next_content = entries[number][1] if number < len(entries) else ""
        child: Any = [] if next_indent > indent and next_content.startswith("- ") else {}
        if not isinstance(parent, dict):
            raise ValueError(f"YAML line {number}: mapping has no mapping parent")
        parent[key] = child
        stack.append((indent, child))
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or ordinary dependency-free YAML configuration."""

    target = Path(path)
    text = target.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _simple_yaml(text)
    if not isinstance(value, dict):
        raise ValueError(f"{target} must contain a mapping")
    return value


def dump_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not an object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def recover_jsonl(path: str | Path) -> bool:
    """Discard only an incomplete final JSONL record after an interrupted append."""

    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        return False
    with target.open("rb+") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        handle.seek(end - 1)
        if handle.read(1) == b"\n":
            return False
        cursor = end - 1
        while cursor > 0:
            cursor -= 1
            handle.seek(cursor)
            if handle.read(1) == b"\n":
                cursor += 1
                break
        else:
            cursor = 0
        handle.truncate(cursor)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Durably append one complete episode; recovery handles a torn final line."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with target.open("ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


def append_csv(path: str | Path, fieldnames: list[str], row: dict[str, Any], *, write_header: bool = False) -> None:
    """Durably append one CSV row using the same dialect as ``write_csv``."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
