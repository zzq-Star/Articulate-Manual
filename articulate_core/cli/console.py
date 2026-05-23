import sys

from rich.console import Console as RichConsole


def create_console() -> RichConsole:
    """Create a Rich Console configured for cross-platform compatibility."""
    # On Windows, reconfigure stdout to UTF-8 to avoid GBK encoding errors
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    return RichConsole()
