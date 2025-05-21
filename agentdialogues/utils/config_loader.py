"""Configuration loader."""

from pathlib import Path
from typing import Any, Optional

import tomllib


def load_agentdialogues_config(
    filename: str = "agentdialogues.toml",
    override_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Load the agentdialogues config file in TOML format.

    Priority:
    1. If `override_path` is provided, load from there.
    2. Otherwise, look for `agentdialogues.toml` in the current working directory.
    3. If not found, fall back to the default config file bundled with the agentdialogues project root.

    Args:
        filename (str): Name of the config file (default: 'agentdialogues.toml').
        override_path (Path | None): Explicit path to a config file.

    Returns:
        dict[str, Any]: Parsed config as a dictionary.

    Raises:
        FileNotFoundError: If no config file is found.
        tomllib.TOMLDecodeError: If the TOML file cannot be parsed.
    """
    possible_paths = []

    if override_path:
        possible_paths.append(Path(override_path).expanduser().resolve())
    else:
        # 1. Check in current working directory (user override)
        possible_paths.append(Path.cwd() / filename)

        # 2. Fallback: look in the root of the agentdialogues module
        possible_paths.append(Path(__file__).parent.parent / filename)

    for path in possible_paths:
        if path.exists():
            with path.open("rb") as f:
                return tomllib.load(f)

    raise FileNotFoundError(f"No config file found in: {possible_paths}")
