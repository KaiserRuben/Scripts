"""Output writers."""
from .markdown import save_markdown
from .csv import save_csv
from .folder import save_folder

__all__ = ["save_markdown", "save_csv", "save_folder"]
