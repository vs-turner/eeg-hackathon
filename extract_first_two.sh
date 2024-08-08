#!/bin/bash

# Directory containing the subject folders
FILES_DIR="files"
DEST_DIR="new_data"

mkdir -p "$DEST_DIR"

# Loop through each subject folder in the files directory
for subject_dir in "$FILES_DIR"/*; do
    if [ -d "$subject_dir" ]; then # if directory
        # Extract R01.edf and R02.edf for each subject
        cp $subject_dir/S*01.edf $DEST_DIR
        cp $subject_dir/S*02.edf $DEST_DIR
    fi
done
