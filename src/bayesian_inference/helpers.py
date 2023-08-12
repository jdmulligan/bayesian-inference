"""General helpers + utilities

Adapted from `mammoth`.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import rich
import rich.progress
from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

# We need a consistent console object to set everything up properly
# (Namely, logging with the progress bars), so we define it here.
rich_console = Console()



class RichModuleNameHandler(RichHandler):
    """Renders the module name instead of the log path."""

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Optional[rich.traceback.Traceback],
        message_renderable: "rich.console.ConsoleRenderable",
    ) -> "rich.console.ConsoleRenderable":
        """Render log for display.

        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.
        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        # START modifications (originally for STAT)
        path = record.name
        # END modifications
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )
        return log_renderable  # noqa: RET504


def progress_bar() -> rich.progress.Progress:
    """Define rich progress bar with my preferred parameters"""
    return rich.progress.Progress(
        # The first four columns are from:
        # `*Progress.get_default_columns()`
        # However, I wanted a wider BarColumn, so I coped the definitions and modified the width
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(bar_width=None),
        rich.progress.TaskProgressColumn(),
        rich.progress.TimeRemainingColumn(),
        rich.progress.MofNCompleteColumn(),
        "Elapsed:",
        rich.progress.TimeElapsedColumn(),
        console=rich_console,
        refresh_per_second=1, speed_estimate_period=30,
        expand=True,
    )


def setup_logging(
    level: int = logging.INFO,
) -> bool:
    """Configure logging.

    Args:
        level: Logging level. Default: "INFO".

    Returns:
        True if logging was set up successfully.
    """
    # Setup logging
    FORMAT = "%(message)s"
    # NOTE: For shutting up parsl, it's import to set the level both in the logging setup
    #       as well as in the handler.
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichModuleNameHandler(level=level, console=rich_console, rich_tracebacks=True)],
    )

    # Capture warnings into logging
    logging.captureWarnings(True)

    # Quiet down some loggers for sanity
    # None for now...

    return True
