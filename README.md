### findfat

a simple cli tool to find what's eating your disk space on linux or macos.

it scans directories recursively and shows you the biggest files and folders, sorted by size by default.

### #install

run `uv pip install .` (or `pip install .`)

that's it, the `findfat` command should now be available.

### usage examples

**scan current directory (.) for top 15 items >= 1mb (defaults)**
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

**sort by modification time (most recent first) instead of size**
```bash
findfat . -s mtime
```

**sort by access time**
```bash
findfat . -s atime
```

**exclude node_modules and .git directories (can use multiple -e)**
```bash
findfat ~/code -e node_modules -e .git -e build -e dist
```

**exclude log files**
```bash
findfat /var/log -e '*.log'
```

**follow symbolic links (use carefully! might scan forever or outside target area)**
```bash
findfat /some/path -l
```

**output as json instead of a table (good for scripting)**
```bash
findfat /tmp -m 1k -n 50 -o json
```

**pipe json output to jq for further processing (get the single largest item)**
```bash
findfat . -o json | jq '.[0]'
```

**disable the progress bar**
```bash
findfat /very/deep/path --no-progress
```

**get help**
```bash
findfat --help
```