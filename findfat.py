# src/main.py
# --- Core Libraries ---
import argparse
import json
import math
import os
import stat as stat_module # Avoid conflict with os.stat
import sys
import re
import fnmatch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import structlog

# --- Dependencies (Install with: uv pip install rich structlog) ---
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
    from rich.table import Table
    from rich.console import Console
    from rich.live import Live
    from rich import print as rich_print # Use rich's print for better formatting
    _RICH_AVAILABLE = True
except ImportError:
    rich_print = print # Fallback to standard print
    _RICH_AVAILABLE = False
    # Define dummy classes if rich is not available to avoid NameErrors
    class Progress:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def add_task(self, *args, **kwargs): return 0
        def update(self, *args, **kwargs): pass
        def stop(self): pass
    class Table: pass
    class Console:
         def __init__(self, *args, **kwargs): pass
         def print(self, *args, **kwargs): rich_print(*args, **kwargs) # Use standard print
    class Live:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args, **kwargs): pass

# --- Constants ---
# Define stat functions/constants locally to avoid re-importing stat_module everywhere
S_ISREG = stat_module.S_ISREG
S_ISDIR = stat_module.S_ISDIR
S_ISLNK = stat_module.S_ISLNK

# --- structlog Configuration ---
# (Structlog configuration remains the same as before - omitted here for brevity)
def configure_logging(log_level_name: str = "INFO"):
    """Configures structlog logging based on desired level."""
    log_level = getattr(structlog.stdlib.logging, log_level_name.upper(), structlog.stdlib.logging.INFO)

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
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
    )

    handler = structlog.stdlib.logging.StreamHandler(sys.stderr) # Log to stderr
    handler.setFormatter(formatter)

    root_logger = structlog.stdlib.logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

log = structlog.get_logger("findfat")
# --- End structlog Configuration ---


# --- Helper Functions ---
def parse_size(size_str: str) -> int:
    """Parses human-readable size string (e.g., '100M', '2G', '1.5T') into bytes."""
    size_str = size_str.strip().upper()
    if not size_str:
        return 0

    # Regex to capture the numeric part and the unit (optional)
    # - Allows integer or decimal numbers (e.g., 100, 1.5)
    # - Captures the unit (K, M, G, T, P) optionally followed by 'B' or 'IB'
    match = re.match(r"^(\d+(\.\d+)?)\s*([KMGTPI]?)B?$", size_str)
    if not match:
        # Handle case where it's just a number (bytes)
        if size_str.replace(".", "", 1).isdigit():
            try:
                return int(float(size_str))
            except ValueError:
                # Should not happen if isdigit passed, but safety check
                raise ValueError(f"Invalid numeric format: {size_str!r}")
        else:
            raise ValueError(f"Invalid size format: {size_str!r}")

    num_part_str = match.group(1)  # The numeric part (e.g., "1", "1.5")
    unit = match.group(3)  # The unit character (e.g., "G", "M", or "")

    try:
        num = float(num_part_str)
    except ValueError:
        # Should be caught by regex, but safety check
        raise ValueError(f"Invalid number format: {num_part_str!r} in {size_str!r}")

    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    multiplier = units.get(
        unit, 1
    )  # Default to 1 (bytes) if unit is empty or not found

    return int(num * multiplier)


def human_readable_size(size_bytes: int) -> str:
    """Converts bytes into human-readable string (KiB, MiB, GiB)."""
    if size_bytes is None or size_bytes < 0: return "N/A"
    if size_bytes == 0: return "0 B"
    size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    i = min(int(math.floor(math.log(abs(size_bytes), 1024))) if abs(size_bytes) > 0 else 0, len(size_name) - 1)
    p = math.pow(1024, i)
    s = round(size_bytes / p, 1)
    if s == int(s): s = int(s) # Avoid ".0"
    return f"{s} {size_name[i]}"


def get_path_metadata(path: Path, follow_links: bool, use_lstat_for_size: bool = False) -> dict | None:
    """
    Gets file/dir stats using standard POSIX calls (works on Linux/macOS).
    Handles errors and symlinks. Reports logical size (st_size).
    Note: On filesystems like APFS (macOS), actual disk usage might differ due
    to features like cloning or snapshots. This tool reports logical sizes.
    """
    try:
        # Use lstat if we specifically need link info OR if we don't want to follow links
        # Use stat if we want info about the target of a link
        do_lstat = use_lstat_for_size or not follow_links

        try:
            # Use lstat() or stat() based on follow_links flag. Both are POSIX standard.
            stats = path.lstat() if do_lstat else path.stat()
        except FileNotFoundError:
            # Handle broken symlinks gracefully if follow_links=True
            if not do_lstat:
                 try:
                     # Check if it's actually a link before declaring it broken
                     link_stats = path.lstat()
                     if S_ISLNK(link_stats.st_mode):
                         stats = link_stats # Get info about the link itself
                         log.debug("Using lstat for broken symlink", path=str(path))
                     else:
                         raise # It wasn't a link, re-raise FileNotFoundError
                 except OSError:
                     raise # Failed even getting lstat, re-raise original FileNotFoundError
            else:
                 raise # Re-raise original error if it wasn't a link issue or if we already tried lstat

        # Determine item type using standard stat flags
        mode = stats.st_mode
        is_dir = S_ISDIR(mode)
        is_file = S_ISREG(mode)
        is_link = S_ISLNK(mode)

        # Get size: st_size is logical size (standard across platforms).
        # For directories, st_size meaning varies; we calculate dir size later.
        # For symlinks with lstat, st_size is path length.
        # We prioritize regular file size or link size (if using lstat).
        size = stats.st_size if is_file or (is_link and do_lstat) else 0

        # st_blocks gives allocation size in 512-byte blocks (more relevant for actual disk usage)
        # We'll include it but primarily sort/filter by logical size (st_size) for consistency with tools like 'ls -l'.
        # Note: macOS might report 0 st_blocks for some items depending on APFS features.
        blocks = getattr(stats, 'st_blocks', 0)
        allocated_size = blocks * 512 if blocks > 0 else None # Provide calculated allocated size

        return {
            "path": path,
            "size": size, # Logical size (primary focus)
            "allocated_size": allocated_size, # Filesystem allocation size (secondary info)
            "is_dir": is_dir,
            "is_file": is_file,
            "is_link": is_link,
            "atime": datetime.fromtimestamp(stats.st_atime),
            "mtime": datetime.fromtimestamp(stats.st_mtime),
            "ctime": datetime.fromtimestamp(stats.st_ctime), # Note: ctime varies (inode change Linux vs creation macOS)
            "uid": stats.st_uid,
            "gid": stats.st_gid,
            "inode": stats.st_ino,
            "dev": stats.st_dev,
            "nlink": stats.st_nlink,
            "mode": mode,
        }
    except OSError as e:
        # Catching OSError covers PermissionError, FileNotFoundError etc.
        log.warning("Failed to stat path", path=str(path), error=str(e), exc_info=False)
        return None
    except Exception as e:
        # Catch unexpected errors during stat
        log.error("Unexpected error getting metadata", path=str(path), error=str(e), exc_info=True)
        return None

# (should_exclude remains the same)
def should_exclude(path: Path, base_path: Path, exclude_patterns: list[str]) -> bool:
    """Check if a path matches any exclusion patterns."""
    if not exclude_patterns:
        return False
    relative_path_str = str(path.relative_to(base_path)) if path.is_absolute() and path.is_relative_to(base_path) else str(path)
    path_str = str(path)
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path.name, pattern):
            log.debug("Excluding path based on name match", path=path_str, pattern=pattern)
            return True
        if fnmatch.fnmatch(relative_path_str, pattern):
            log.debug("Excluding path based on relative path match", path=relative_path_str, pattern=pattern)
            return True
        if fnmatch.fnmatch(path_str, pattern):
             log.debug("Excluding path based on full path match", path=path_str, pattern=pattern)
             return True
    return False


# --- Main Scan Logic ---
# (find_fat core logic remains the same - uses the updated get_path_metadata)
# Key aspect: Uses os.walk, Path objects, and stat results, all cross-platform.
def find_fat(
    scan_path_str: str,
    top_n: int = 10,
    min_size_bytes: int = 1 * 1024 * 1024,
    item_type: str = "both",
    sort_by: str = "size",
    follow_links: bool = False,
    exclude_patterns: list[str] | None = None,
    show_progress: bool = True,
    output_format: str = "table",
):
    """Scans a path recursively to find large files and directories."""
    scan_path = Path(scan_path_str).resolve()
    if not scan_path.is_dir():
        log.error("Scan path does not exist or is not a directory", path=str(scan_path))
        sys.exit(1)

    exclude_patterns = exclude_patterns or []
    log.info(
        "Starting scan",
        path=str(scan_path),
        min_size=human_readable_size(min_size_bytes),
        top_n=top_n,
        type=item_type,
        sort_by=sort_by,
        follow_links=follow_links,
        exclusions=exclude_patterns if exclude_patterns else "None",
        output=output_format,
    )

    console = Console(stderr=True) # Progress to stderr
    results = []
    dir_sizes: dict[Path, int] = defaultdict(int)
    processed_dev_inodes = set() # Track (device, inode) for hard links
    error_count = 0
    items_processed = 0

    progress_columns = [
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TaskProgressColumn(), TimeElapsedColumn(), TextColumn("{task.fields[path]}"),
    ] if _RICH_AVAILABLE else []

    progress = Progress(*progress_columns, console=console, transient=False, disable=not show_progress or not _RICH_AVAILABLE)

    try:
        with progress:
            scan_task = progress.add_task("Scanning...", total=None, path=str(scan_path))

            # os.walk is efficient and cross-platform
            for dirpath_str, dirnames_orig, filenames_orig in os.walk(str(scan_path), topdown=False, followlinks=follow_links, onerror=lambda e: log.warning("Permission denied or walk error", path=getattr(e, 'filename', '?'), error=str(e))):

                current_dir_path = Path(dirpath_str)
                progress.update(scan_task, description="Scanning", path=f".../{current_dir_path.name}")

                # Filter based on exclusions
                dirnames = [d for d in dirnames_orig if not should_exclude(current_dir_path / d, scan_path, exclude_patterns)]
                filenames = [f for f in filenames_orig if not should_exclude(current_dir_path / f, scan_path, exclude_patterns)]

                current_dir_calculated_size = 0

                # Process files
                for filename in filenames:
                    items_processed += 1
                    file_path = current_dir_path / filename
                    use_lstat_size_for_item = not follow_links # Match stat/lstat to follow_links intent

                    metadata = get_path_metadata(file_path, follow_links=follow_links, use_lstat_for_size=use_lstat_size_for_item)
                    if metadata is None:
                        error_count += 1
                        continue

                    # Use logical size for filtering and primary reporting
                    item_size = metadata['size']

                    # Handle hard links using (dev, inode)
                    if metadata['is_file'] and metadata['nlink'] > 1:
                        dev_inode = (metadata['dev'], metadata['inode'])
                        if dev_inode in processed_dev_inodes:
                            log.debug("Skipping hard link already counted", path=str(file_path), inode=metadata['inode'])
                            continue
                        processed_dev_inodes.add(dev_inode)

                    current_dir_calculated_size += item_size # Add logical size to dir total

                    # Add file to results if it meets criteria (based on logical size)
                    if (item_type == "files" or item_type == "both") and item_size >= min_size_bytes:
                         metadata['type'] = 'file'
                         results.append(metadata)
                         log.debug("Found large file", path=str(file_path), size=human_readable_size(item_size))

                # Add sizes of included subdirectories
                for dirname in dirnames:
                    subdir_path = current_dir_path / dirname
                    if subdir_path in dir_sizes: # Check if subdir was processed (not excluded/errored)
                        current_dir_calculated_size += dir_sizes[subdir_path]

                # Store total calculated *logical* size for the directory
                dir_sizes[current_dir_path] = current_dir_calculated_size

                # Add directory to results if it meets criteria
                if item_type == "dirs" or item_type == "both":
                    dir_metadata = get_path_metadata(current_dir_path, follow_links=False) # Get dir's own metadata
                    if dir_metadata and current_dir_calculated_size >= min_size_bytes:
                        dir_metadata['size'] = current_dir_calculated_size # Use calculated logical size
                        dir_metadata['type'] = 'dir'
                        # Fetch allocated size for the dir entry itself (usually small) if needed
                        # dir_metadata['allocated_size'] = dir_metadata.get('allocated_size')
                        results.append(dir_metadata)
                        log.debug("Found large directory", path=str(current_dir_path), calculated_size=human_readable_size(current_dir_calculated_size))

                progress.update(scan_task, advance=1) # Advance progress per directory processed

            progress.update(scan_task, description="Scan complete.", path="")

    except KeyboardInterrupt:
        log.warning("Scan interrupted by user.")
        progress.stop()
        sys.exit(1)
    finally:
        progress.stop()

    # --- Post-processing ---
    log.info(f"Scan finished. Processed ~{items_processed} items.", found=len(results), errors=error_count)
    if not results:
        rich_print("[yellow]No items found matching the criteria.[/yellow]")
        return

    # Sort results (allow sorting by allocated_size too)
    valid_sort_keys = ["size", "allocated_size", "mtime", "atime", "ctime"]
    if sort_by not in valid_sort_keys:
        log.warning(f"Invalid sort key '{sort_by}'. Defaulting to 'size'. Valid keys: {valid_sort_keys}")
        sort_by = "size"

    # Handle potentially missing keys (like allocated_size) during sort
    sort_key_func = lambda item: item.get(sort_by) if isinstance(item.get(sort_by), (int, float, datetime)) else (datetime.min if isinstance(item.get(sort_by), datetime) else 0)

    try:
        log.debug(f"Sorting {len(results)} results by '{sort_by}' descending")
        results.sort(key=sort_key_func, reverse=True) # Default all sorts to descending
    except TypeError as e:
        log.error(f"Failed to sort results by '{sort_by}'. Falling back to size.", error=str(e))
        results.sort(key=lambda item: item.get('size', 0), reverse=True)

    top_results = results[:top_n]

    # --- Output ---
    output_items(top_results, output_format, scan_path)


# (output_items adjusted slightly for allocated_size)
def output_items(items: list[dict], format: str, scan_path: Path):
    """Formats and prints the results."""
    console = Console() # Output to stdout

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
                elif k == 'mode': # Optionally convert mode to octal string
                    new_item[k] = oct(v)
                else:
                    new_item[k] = v
            serializable_items.append(new_item)
        console.print(json.dumps(serializable_items, indent=2))

    elif format == "table":
        if not _RICH_AVAILABLE:
            # Basic fallback table (unchanged)
            rich_print("\n--- Top Items (rich not installed, basic table) ---")
            # ... (omitted for brevity) ...
            return

        # Rich Table - Add Allocated Size column
        table = Table(title=f"Top {len(items)} Largest Items from {scan_path}", show_header=True, header_style="bold magenta")
        table.add_column("Type", style="dim", width=4)
        table.add_column("Log. Size", style="green", justify="right") # Logical Size
        table.add_column("Alloc. Size", style="cyan", justify="right") # Allocated Size
        table.add_column("Modified", style="yellow")
        table.add_column("Accessed", style="magenta") # Changed color slightly
        table.add_column("Path", style="bold blue", max_width=console.width // 2, overflow="fold")

        for item in items:
            path_obj = item['path']
            try:
                display_path = path_obj.relative_to(scan_path)
                display_path_str = str(display_path) if not str(display_path).startswith('..') else str(path_obj)
            except ValueError:
                display_path_str = str(path_obj)

            table.add_row(
                item.get('type', '?'),
                human_readable_size(item.get('size', -1)),
                human_readable_size(item.get('allocated_size')), # Display allocated size
                item.get('mtime', datetime.min).strftime('%Y-%m-%d %H:%M'),
                item.get('atime', datetime.min).strftime('%Y-%m-%d %H:%M'),
                display_path_str,
            )
        console.print(table)
        console.print("[dim]Note: Log. Size = Logical size (st_size). Alloc. Size = Space allocated on disk (st_blocks * 512). May differ due to sparse files, block sizes, or FS features (e.g., APFS clones).[/dim]")

    else:
        log.error(f"Unknown output format: {format}")

# --- Argparse Setup & Main Execution ---
# (main function updated slightly for sort_by options)
def main():
    parser = argparse.ArgumentParser(
        description="Find large files and directories ('fat') recursively on Linux/macOS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Reports logical file sizes. Actual disk usage may vary on some filesystems (e.g. APFS)."
    )

    parser.add_argument(
        "path",
        help="The directory path to start scanning.",
    )
    parser.add_argument(
        "-n", "--top", type=int, default=15,
        help="Show the top N largest items.",
    )
    parser.add_argument(
        "-m", "--min-size", type=str, default="1M",
        help="Minimum logical size to report (e.g., 500K, 100M, 2G).",
    )
    parser.add_argument(
        "-t", "--type", choices=["files", "dirs", "both"], default="both",
        help="Type of items to report.",
    )
    parser.add_argument(
        "-s", "--sort-by",
        choices=["size", "allocated_size", "mtime", "atime", "ctime"], # Added allocated_size
        default="size",
        help="Sort results by: 'size' (logical), 'allocated_size' (disk usage), 'mtime', 'atime', 'ctime'. All sorted descending.",
    )
    parser.add_argument(
        "-L", "--follow-links", action="store_true",
        help="Follow symbolic links. Use with caution.",
    )
    parser.add_argument(
        "-e", "--exclude", action="append", metavar="PATTERN",
        help="Exclude paths matching PATTERN (wildcards ok). Multiple allowed.",
    )
    parser.add_argument(
        "-o", "--output-format", choices=["table", "json"], default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--no-progress", action="store_false", dest="show_progress",
        help="Disable the progress bar.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_const", dest="log_level", const="DEBUG", default="INFO",
        help="Enable verbose (DEBUG level) logging to stderr.",
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

    find_fat(
        scan_path_str=args.path,
        top_n=args.top,
        min_size_bytes=min_size_bytes,
        item_type=args.type,
        sort_by=args.sort_by,
        follow_links=args.follow_links,
        exclude_patterns=args.exclude,
        show_progress=args.show_progress,
        output_format=args.output_format,
    )

if __name__ == "__main__":
    main()