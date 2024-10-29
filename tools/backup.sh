#! /usr/bin/bash
# backup.sh - A dumb script to run an arbitrary command if a directory exists
# You can set this up as a cron task to automate backups of training checkpoints
#
# E.g. */n * * * * /path/to/backup.sh /path/to/directory /path/to/backup/command
# Where n is the interval in minutes

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <backup command>"
    exit 1
fi

DIRECTORY=$1
COMMAND=$2

# Check if the directory exists
if [ -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY exists. Running command: $COMMAND"
    $COMMAND
else
    echo "Directory $DIRECTORY does not exist."
    exit 1
fi
