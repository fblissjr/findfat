### findfat

a simple cli tool to find what's eating your disk space on linux or macos.

it scans directories recursively and shows you the biggest files and folders with pretty structured outputs and sorting options

### #install

run `uv pip install .` (or `pip install .`)

that's it, the `findfat` command should now be available.

### usage examples

**scan current directory (.) for top 15 items >= 1mb (defaults: sorted by size descending)**
```bash
findfat .
```

**scan a specific path**
```bash
findfat /var/log
```

**find top 5 biggest things in your downloads folder**
```bash
findfat ~/downloads -n 5
```

**find items bigger than 500 megabytes in /data**
```bash
findfat /data -m 500m
```

**find only *files* bigger than 10 megabytes**
```bash
findfat /home/user/projects -t files -m 10m
```

**find only *directories* bigger than 1 gigabyte**
```bash
findfat /mnt/storage -t dirs -m 1g
```

**sort by modification time (default: ascending - oldest first)**
```bash
findfat . -s mtime
```

**sort by modification time descending (newest first)**
```bash
findfat . -s mtime -d desc
```

**sort by access time (default: ascending - oldest first)**
```bash
findfat . -s atime
```

**sort by access time descending (newest first)**
```bash
findfat . -s atime -d desc```

**sort by allocated size (default: descending - largest allocation first)**
```bash
findfat . -s allocated_size
```

**sort by logical size ascending (smallest first)**
```bash
findfat . -s size -d asc
```

**exclude node_modules and .git directories (can use multiple -e)**
```bash
findfat ~/code -e node_modules -e .git -e build -e dist```

**exclude log files**
```bash
findfat /var/log -e '*.log'
```

**follow symbolic links (use carefully!)**
```bash
findfat /some/path -l
```

**output as json instead of a table**
```bash
findfat /tmp -m 1k -n 50 -o json
```

**pipe json output to jq (get the single largest item)**
```bash
findfat . -o json | jq '.[0]'
```

**disable the progress bar**
```bash
findfat /very/deep/path --no-progress
```

**get help (shows all options and defaults)**
```bash
findfat --help
```