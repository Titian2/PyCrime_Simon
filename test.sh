#!/bin/zsh

directory="test"
parent_directory_name=$(basename "$directory")
find "$directory" -type f \( -name 'Grouped_Output_*' -a -name '*.txt' -o -name 'AverageLoss_*' -a -name '*.png' \) -print | zip "$parent_directory_name.zip" -@
