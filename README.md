# findfat

a simple cli tool to find what's eating your disk space on linux or macos.

it scans directories recursively and shows you the biggest files and folders, sorted by size by default.

## install

run `uv pip install .` (or `pip install .`)

that's it, the `findfat` command should now be available.

## usage examples

# scan current directory (.) for top 15 items >= 1mb (defaults)
findfat .

# scan a specific path
findfat /var/log

# find top 5 biggest things in your downloads folder
findfat ~/downloads -n 5

# find items bigger than 500 megabytes in /data
findfat /data -m 500m

# find only *files* bigger than 10 megabytes
findfat /home/user/projects -t files -m 10m

# find only *directories* bigger than 1 gigabyte
findfat /mnt/storage -t dirs -m 1g

# sort by modification time (most recent first) instead of size
findfat . -s mtime

# sort by access time
findfat . -s atime

# exclude node_modules and .git directories (can use multiple -e)
findfat ~/code -e node_modules -e .git -e build -e dist

# exclude log files
findfat /var/log -e '*.log'

# follow symbolic links (use carefully! might scan forever or outside target area)
findfat /some/path -l

# output as json instead of a table (good for scripting)
findfat /tmp -m 1k -n 50 -o json

# pipe json output to jq for further processing
findfat . -o json | jq '.[0]' # get the single largest item as json

# disable the progress bar
findfat /very/deep/path --no-progress

# get help
findfat --help