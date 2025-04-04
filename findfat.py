# src/main.py
# --- Core Libraries ---
import argparse
import json
import math
import os
import re  # For parse_size
import stat as stat_module
import sys
from datetime import datetime, timezone
import fnmatch
from collections import defaultdict
from pathlib import Path

# --- Dependencies ---
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
    from rich.table import Table
    from rich.console import Console
    from rich.live import Live
    from rich import print as rich_print
    _RICH_AVAILABLE = True
except ImportError:
    # Rich fallback... (omitted for brevity)
    rich_print = print
    _RICH_AVAILABLE = False

    class Progress:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def add_task(self, *args, **kwargs): return 0
        def update(self, *args, **kwargs): pass
        def stop(self): pass

    class Table:
        pass  # type: ignore

    class Console:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def print(self, *args, **kwargs):
            rich_print(*args, **kwargs)

    class Live:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args, **kwargs): pass


import structlog

# --- Constants ---
S_ISREG = stat_module.S_ISREG
S_ISDIR = stat_module.S_ISDIR
S_ISLNK = stat_module.S_ISLNK

# --- Default Sort Directions ---
DEFAULT_SORT_DIRECTIONS = {
    "size": "desc",
    "allocated_size": "desc",
    "mtime": "asc",  # Default: Oldest modified first
    "atime": "asc",  # Default: Oldest accessed first
    "ctime": "asc",  # Default: Oldest created/changed first
}


# --- structlog Configuration ---
def configure_logging(log_level_name: str = "INFO"):
    log_level = getattr(
        structlog.stdlib.logging, log_level_name.upper(), structlog.stdlib.logging.INFO
    )
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
    ]
    renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
    )
    handler = structlog.stdlib.logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger = structlog.stdlib.logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


log = structlog.get_logger("findfat")


def parse_size(size_str: str) -> int:
    """Parses human-readable size string (e.g., '100M', '2G', '1.5T') into bytes."""
    size_str = size_str.strip().upper()
    if not size_str:
        return 0
    match = re.match(r"^(\d+(\.\d+)?)\s*([KMGTPI]?)B?$", size_str)
    if not match:
        if size_str.replace(".", "", 1).isdigit():
            try:
                return int(float(size_str))
            except ValueError:
                raise ValueError(f"Invalid numeric format: {size_str!r}")
        else:
            raise ValueError(f"Invalid size format: {size_str!r}")
    num_part_str, _, unit = match.groups()
    try:
        num = float(num_part_str)
    except ValueError:
        raise ValueError(f"Invalid number format: {num_part_str!r} in {size_str!r}")
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    multiplier = units.get(unit, 1)
    return int(num * multiplier)

def human_readable_size(size_bytes: int | None) -> str:
    """Converts bytes into human-readable string (KiB, MiB, GiB)."""
    if size_bytes is None or size_bytes < 0: return "N/A"
    if size_bytes == 0: return "0 B"
    size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    i = min(int(math.floor(math.log(abs(size_bytes), 1024))) if abs(size_bytes) > 0 else 0, len(size_name) - 1)
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    if s == int(s):
        s = int(s)
    return f"{s} {size_name[i]}"


def get_path_metadata(
    path: Path, follow_links: bool, use_lstat_for_size: bool = False
) -> dict | None:
    """Gets file/dir stats using standard POSIX calls (works on Linux/macOS)."""
    try:
        do_lstat = use_lstat_for_size or not follow_links
        try:
            stats = path.lstat() if do_lstat else path.stat()
        except FileNotFoundError:
            if not do_lstat:
                try:
                    link_stats = path.lstat()
                    if S_ISLNK(link_stats.st_mode):
                        stats = link_stats
                        log.debug("Using lstat for broken symlink", path=str(path))
                    else:
                        raise
                except OSError:
                    raise
            else:
                raise
        mode = stats.st_mode
        is_dir = S_ISDIR(mode)
        is_file = S_ISREG(mode)
        is_link = S_ISLNK(mode)
        size = stats.st_size if is_file or (is_link and do_lstat) else 0
        blocks = getattr(stats, "st_blocks", 0)
        allocated_size = blocks * 512 if blocks > 0 else None
        # Ensure timezone-aware datetimes for correct comparison if needed later
        atime_dt = datetime.fromtimestamp(stats.st_atime, tz=timezone.utc)
        mtime_dt = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)
        ctime_dt = datetime.fromtimestamp(stats.st_ctime, tz=timezone.utc)

        return {
            "path": path,
            "size": size,
            "allocated_size": allocated_size,
            "is_dir": is_dir,
            "is_file": is_file,
            "is_link": is_link,
            "atime": atime_dt,
            "mtime": mtime_dt,
            "ctime": ctime_dt,
            "uid": stats.st_uid,
            "gid": stats.st_gid,
            "inode": stats.st_ino,
            "dev": stats.st_dev,
            "nlink": stats.st_nlink,
            "mode": mode,
        }
    except OSError as e:
        log.warning("Failed to stat path", path=str(path), error=str(e), exc_info=False)
        return None
    except Exception as e:
        log.error(
            "Unexpected error getting metadata",
            path=str(path),
            error=str(e),
            exc_info=True,
        )
        return None


def should_exclude(path: Path, base_path: Path, exclude_patterns: list[str]) -> bool:
    """Check if a path matches any exclusion patterns."""
    if not exclude_patterns:
        return False
    try:
        relative_path_str = str(path.relative_to(base_path))
    except ValueError:
        relative_path_str = str(path)
    path_str = str(path)
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path.name, pattern):
            log.debug("Excluding based on name", path=path_str, pattern=pattern)
            return True
        if fnmatch.fnmatch(relative_path_str, pattern):
            log.debug(
                "Excluding based on relative path",
                path=relative_path_str,
                pattern=pattern,
            )
            return True
    return False


def get_sort_key_value(item: dict, sort_by: str):
    """Gets the value from an item dict for sorting, handling missing keys and types."""
    value = item.get(sort_by)
    if isinstance(value, (int, float, datetime)):
        return value
    else:
        # Provide a default value suitable for comparison if key is missing or wrong type
        if sort_by in ["mtime", "atime", "ctime"]:
            # Use earliest possible datetime for sorting missing dates
            return datetime.min.replace(tzinfo=timezone.utc)
        else:
            # Use 0 for missing sizes or other numeric fields
            return 0


def find_fat(
    scan_path_str: str,
    top_n: int,
    min_size_bytes: int,
    item_type: str,
    sort_by: str,
    sort_direction: str,
    follow_links: bool,
    exclude_patterns: list[str] | None,
    show_progress: bool,
    output_format: str,
):
    """Scans a path recursively to find large files and directories."""
    scan_path = Path(scan_path_str).resolve()
    if not scan_path.is_dir():
        log.error("Scan path is not a directory", path=str(scan_path))
        sys.exit(1)

    exclude_patterns = exclude_patterns or []
    log.info(
        "Starting scan",
        path=str(scan_path),
        min_size=human_readable_size(min_size_bytes),
        top_n=top_n,
        type=item_type,
        sort_by=sort_by,
        sort_direction=sort_direction,
        follow_links=follow_links,
        exclusions=exclude_patterns if exclude_patterns else "None",
        output=output_format,
    )

    console = Console(stderr=True)
    results = []
    dir_sizes: dict[Path, int] = defaultdict(int)
    processed_dev_inodes = set()
    error_count = 0
    items_processed = 0
    progress_columns = (
        [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[path]}"),
        ]
        if _RICH_AVAILABLE
        else []
    )
    progress = Progress(
        *progress_columns,
        console=console,
        transient=False,
        disable=not show_progress or not _RICH_AVAILABLE,
    )

    try:
        with progress:
            scan_task = progress.add_task(
                "Scanning...", total=None, path=str(scan_path)
            )
            for dirpath_str, dirnames_orig, filenames_orig in os.walk(
                str(scan_path),
                topdown=False,
                followlinks=follow_links,
                onerror=lambda e: log.warning(
                    "Permission denied or walk error",
                    path=getattr(e, "filename", "?"),
                    error=str(e),
                ),
            ):
                current_dir_path = Path(dirpath_str)
                progress.update(
                    scan_task,
                    description="Scanning",
                    path=f".../{current_dir_path.name}",
                )
                dirnames = [
                    d
                    for d in dirnames_orig
                    if not should_exclude(
                        current_dir_path / d, scan_path, exclude_patterns
                    )
                ]
                filenames = [
                    f
                    for f in filenames_orig
                    if not should_exclude(
                        current_dir_path / f, scan_path, exclude_patterns
                    )
                ]
                current_dir_calculated_size = 0
                for filename in filenames:
                    items_processed += 1
                    file_path = current_dir_path / filename
                    use_lstat_size = not follow_links
                    metadata = get_path_metadata(
                        file_path,
                        follow_links=follow_links,
                        use_lstat_for_size=use_lstat_size,
                    )
                    if metadata is None:
                        error_count += 1
                        continue
                    item_size = metadata["size"]
                    if metadata["is_file"] and metadata["nlink"] > 1:
                        dev_inode = (metadata["dev"], metadata["inode"])
                        if dev_inode in processed_dev_inodes:
                            log.debug("Skipping hard link", path=str(file_path))
                            continue
                        processed_dev_inodes.add(dev_inode)
                    current_dir_calculated_size += item_size
                    if (
                        item_type == "files" or item_type == "both"
                    ) and item_size >= min_size_bytes:
                        metadata["type"] = "file"
                        results.append(metadata)
                        log.debug(
                            "Found large file",
                            path=str(file_path),
                            size=human_readable_size(item_size),
                        )
                for dirname in dirnames:
                    if (subdir_path := current_dir_path / dirname) in dir_sizes:
                        current_dir_calculated_size += dir_sizes[subdir_path]
                dir_sizes[current_dir_path] = current_dir_calculated_size
                if item_type == "dirs" or item_type == "both":
                    dir_metadata = get_path_metadata(
                        current_dir_path, follow_links=False
                    )
                    if dir_metadata and current_dir_calculated_size >= min_size_bytes:
                        dir_metadata["size"] = current_dir_calculated_size
                        dir_metadata["type"] = "dir"
                        results.append(dir_metadata)
                        log.debug(
                            "Found large directory",
                            path=str(current_dir_path),
                            size=human_readable_size(current_dir_calculated_size),
                        )
                progress.update(scan_task, advance=1)
            progress.update(scan_task, description="Scan complete.", path="")
    except KeyboardInterrupt:
        log.warning("Scan interrupted.")
        progress.stop()
        sys.exit(1)
    finally:
        progress.stop()

    log.info(
        f"Scan finished. Processed ~{items_processed} items.",
        found=len(results),
        errors=error_count,
    )
    if not results:
        rich_print("[yellow]No items found matching the criteria.[/yellow]")
        return

    valid_sort_keys = ["size", "allocated_size", "mtime", "atime", "ctime"]
    if sort_by not in valid_sort_keys:
        log.warning(
            f"Invalid sort key '{sort_by}'. Defaulting to 'size'. Valid keys: {valid_sort_keys}"
        )
        sort_by = "size"
        sort_direction = DEFAULT_SORT_DIRECTIONS[sort_by]

    reverse_sort = sort_direction == "desc"
    log.debug(
        f"Sorting {len(results)} results by '{sort_by}' {'descending' if reverse_sort else 'ascending'}"
    )

    try:
        # Use the helper function for the key
        results.sort(
            key=lambda item: get_sort_key_value(item, sort_by), reverse=reverse_sort
        )
    except TypeError as e:
        log.error(
            f"Failed to sort results by '{sort_by}'. Falling back to size descending.",
            error=str(e),
        )
        results.sort(key=lambda item: get_sort_key_value(item, "size"), reverse=True)

    top_results = results[:top_n]

    output_items(top_results, output_format, scan_path)


def output_items(items: list[dict], format: str, scan_path: Path):
    """Formats and prints the results."""
    console = Console()
    if format == "json":
        serializable_items = []
        for item in items:
            new_item = {}
            for k, v in item.items():
                if isinstance(v, Path):
                    try:
                        new_item[k] = str(v.relative_to(scan_path))
                    except ValueError:
                        new_item[k] = str(v)
                elif isinstance(v, datetime):
                    new_item[k] = v.isoformat()
                elif k == "mode":
                    new_item[k] = oct(v)
                else:
                    new_item[k] = v
            serializable_items.append(new_item)
        console.print(json.dumps(serializable_items, indent=2))
    elif format == "table":
        if not _RICH_AVAILABLE:
            rich_print("\n--- Top Items (rich not installed, basic table) ---")
            rich_print(
                f"{'Type':<5} | {'Log. Size':>10} | {'Alloc. Size':>11} | {'Modified':<20} | {'Accessed':<20} | {'Path'}"
            )
            rich_print("-" * 90)
            for item in items:
                path_str = str(item.get("path", "N/A"))
                mtime_str = item.get("mtime", datetime.min).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                atime_str = item.get("atime", datetime.min).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                rich_print(
                    f"{item.get('type', '?'):<5} | "
                    f"{human_readable_size(item.get('size', -1)):>10} | "
                    f"{human_readable_size(item.get('allocated_size')):>11} | "
                    f"{mtime_str:<20} | "
                    f"{atime_str:<20} | "
                    f"{path_str}"
                )
            return

        table = Table(
            title=f"Top {len(items)} Items from {scan_path}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Type", style="dim", width=4)
        table.add_column("Log. Size", style="green", justify="right")
        table.add_column("Alloc. Size", style="cyan", justify="right")
        table.add_column("Modified", style="yellow")
        table.add_column("Accessed", style="magenta")
        table.add_column(
            "Path", style="bold blue", max_width=console.width // 2, overflow="fold"
        )
        for item in items:
            path_obj = item["path"]
            try:
                display_path_str = str(path_obj.relative_to(scan_path))
            except ValueError:
                display_path_str = str(path_obj)
            # Convert stored UTC time back to local time for display if needed
            mtime_local = item.get(
                "mtime", datetime.min.replace(tzinfo=timezone.utc)
            ).astimezone(tz=None)
            atime_local = item.get(
                "atime", datetime.min.replace(tzinfo=timezone.utc)
            ).astimezone(tz=None)
            table.add_row(
                item.get("type", "?"),
                human_readable_size(item.get("size", -1)),
                human_readable_size(item.get("allocated_size")),
                mtime_local.strftime("%Y-%m-%d %H:%M"),
                atime_local.strftime("%Y-%m-%d %H:%M"),
                display_path_str,
            )
        console.print(table)
        console.print(
            "[dim]Note: Log. Size = Logical size. Alloc. Size = Space allocated. Defaults: Size(desc), Time(asc).[/dim]"
        )
    else:
        log.error(f"Unknown output format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description="Find large files and directories ('fat') recursively on Linux/macOS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Use --sort-direction to control sort order. Times displayed in local timezone.",
    )

    parser.add_argument("path", help="The directory path to start scanning.")
    parser.add_argument(
        "-n", "--top", type=int, default=15, help="Show the top N items."
    )
    parser.add_argument(
        "-m",
        "--min-size",
        type=str,
        default="1M",
        help="Minimum logical size (e.g., 500K, 100M, 2G).",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["files", "dirs", "both"],
        default="both",
        help="Type of items to report.",
    )
    parser.add_argument(
        "-s",
        "--sort-by",
        choices=["size", "allocated_size", "mtime", "atime", "ctime"],
        default="size",
        help="Sort results by: 'size' (default: desc), 'allocated_size' (default: desc), "
        "'mtime' (default: asc), 'atime' (default: asc), 'ctime' (default: asc).",
    )
    parser.add_argument(
        "-d",
        "--sort-direction",
        choices=["asc", "desc"],
        default=None,
        help="Sort direction: 'asc' (ascending) or 'desc' (descending). Default depends on --sort-by.",
    )
    parser.add_argument(
        "-L", "--follow-links", action="store_true", help="Follow symbolic links."
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        metavar="PATTERN",
        help="Exclude paths matching PATTERN.",
    )
    parser.add_argument(
        "-o",
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        help="Disable progress bar.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="log_level",
        const="DEBUG",
        default="INFO",
        help="Enable verbose logging.",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    configure_logging(args.log_level)

    try:
        min_size_bytes = parse_size(args.min_size)
    except ValueError as e:
        log.error(f"Invalid --min-size value: {e}", value=args.min_size)
        sys.exit(1)

    final_sort_direction = args.sort_direction
    if final_sort_direction is None:
        final_sort_direction = DEFAULT_SORT_DIRECTIONS.get(args.sort_by, "desc")
        log.debug(
            f"Using default sort direction '{final_sort_direction}' for sort key '{args.sort_by}'"
        )

    find_fat(
        scan_path_str=args.path,
        top_n=args.top,
        min_size_bytes=min_size_bytes,
        item_type=args.type,
        sort_by=args.sort_by,
        sort_direction=final_sort_direction,
        follow_links=args.follow_links,
        exclude_patterns=args.exclude,
        show_progress=args.show_progress,
        output_format=args.output_format,
    )

if __name__ == "__main__":
    main()