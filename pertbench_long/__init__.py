"""PertBench-Long: retrospective experimental replay for scientific agents.

Importing this package does not load torch, call model APIs, or read private data.
"""

from __future__ import annotations

__all__ = ["__version__", "SCHEMA_VERSION"]
__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"


def __getattr__(name: str):
    if name in {"PublicEpisodeSpec", "PrivateEpisodeSpec"}:
        from pertbench_long.schemas.types import PublicEpisodeSpec, PrivateEpisodeSpec

        mapping = {
            "PublicEpisodeSpec": PublicEpisodeSpec,
            "PrivateEpisodeSpec": PrivateEpisodeSpec,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
