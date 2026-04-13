"""Prompt registry that loads YAML prompt definitions from this package.

Each ``*.yaml`` file in this directory defines one prompt version with
frontmatter-style fields (``name``, ``version``, ``description``,
``template``). The registry is a read-only singleton populated at first
access; tests can call :func:`reload` to force a refresh.

Templates are plain Python ``str.format`` templates. The registry
deliberately does NOT wrap them in a ``PromptTemplate`` — callers already
have their own composition logic and snapshot tests guarantee that the
rendered output is byte-identical to the pre-migration strings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptSpec:
    """Single loaded prompt definition.

    Attributes:
        name: Logical prompt name (e.g. ``intent_classify``).
        version: Version string (e.g. ``v1``).
        description: Short human-readable description.
        template: The raw template string with ``{var}`` placeholders.
        source_path: Absolute path of the YAML file this spec came from.
    """

    name: str
    version: str
    description: str
    template: str
    source_path: Path


class PromptRegistry:
    """Loads and serves prompt templates from YAML files."""

    _instance: "PromptRegistry | None" = None
    _lock = Lock()

    def __init__(self, prompts_dir: Path = _PROMPTS_DIR) -> None:
        """Initialise the registry.

        Args:
            prompts_dir: Directory containing prompt YAML files.
        """
        self._prompts_dir = prompts_dir
        self._by_name: dict[str, dict[str, PromptSpec]] = {}
        self._latest: dict[str, PromptSpec] = {}
        self._load()

    def _load(self) -> None:
        """Scan ``prompts_dir`` and populate in-memory indices."""
        self._by_name.clear()
        self._latest.clear()

        for path in sorted(self._prompts_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise ValueError(f"Prompt file {path} must contain a YAML mapping")
            missing = {"name", "version", "template"} - data.keys()
            if missing:
                raise ValueError(
                    f"Prompt file {path} is missing fields: {sorted(missing)}"
                )
            spec = PromptSpec(
                name=str(data["name"]),
                version=str(data["version"]),
                description=str(data.get("description", "")),
                template=str(data["template"]),
                source_path=path,
            )
            versions = self._by_name.setdefault(spec.name, {})
            if spec.version in versions:
                raise ValueError(
                    f"Duplicate prompt {spec.name}@{spec.version} in {path}"
                )
            versions[spec.version] = spec
            # Latest wins by lexicographic version comparison — "v2" > "v1".
            current_latest = self._latest.get(spec.name)
            if current_latest is None or spec.version > current_latest.version:
                self._latest[spec.name] = spec

        logger.info(
            "PromptRegistry loaded %d prompts from %s",
            len(self._by_name), self._prompts_dir,
        )

    def get(self, name: str, version: str | None = None) -> PromptSpec:
        """Return the :class:`PromptSpec` for ``name`` / ``version``.

        Args:
            name: Logical prompt name.
            version: Specific version, or ``None`` for the latest.

        Returns:
            The matching :class:`PromptSpec`.

        Raises:
            KeyError: When no prompt / version matches.
        """
        if version is None:
            spec = self._latest.get(name)
            if spec is None:
                raise KeyError(f"Unknown prompt: {name}")
            return spec
        versions = self._by_name.get(name, {})
        if version not in versions:
            raise KeyError(f"Unknown prompt version: {name}@{version}")
        return versions[version]

    def render(self, name: str, version: str | None = None, /, **kwargs: object) -> str:
        """Fetch ``name`` and format its template with ``**kwargs``.

        Args:
            name: Logical prompt name.
            version: Specific version, or ``None`` for the latest.
            **kwargs: Template variables.

        Returns:
            The rendered prompt string.
        """
        spec = self.get(name, version)
        return spec.template.format(**kwargs)

    def names(self) -> list[str]:
        """Return all registered prompt names."""
        return sorted(self._by_name.keys())


def _singleton() -> PromptRegistry:
    """Return the process-wide registry, constructing it on first call."""
    with PromptRegistry._lock:
        if PromptRegistry._instance is None:
            PromptRegistry._instance = PromptRegistry()
        return PromptRegistry._instance


def get_prompt(name: str, version: str | None = None) -> PromptSpec:
    """Shortcut for ``_singleton().get(name, version)``."""
    return _singleton().get(name, version)


def render_prompt(name: str, version: str | None = None, /, **kwargs: object) -> str:
    """Shortcut for ``_singleton().render(name, version, **kwargs)``."""
    return _singleton().render(name, version, **kwargs)


def reload() -> None:
    """Force a reload of the registry — intended for tests."""
    with PromptRegistry._lock:
        PromptRegistry._instance = PromptRegistry()
