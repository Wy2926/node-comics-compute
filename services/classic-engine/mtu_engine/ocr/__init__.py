"""Paragraph readings; importing this module loads no model runtime."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    texts: list[str]
