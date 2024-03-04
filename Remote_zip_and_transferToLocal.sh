#!/bin/bash
# Define remote and local directories
remote_user="PyCrime_machine"
remote_host="controls"
remote_base_dir="/mnt/data/CRIME/results"
local_base_dir="/Users/simon/Desktop/CRIME_BACKUP"

# Define serial numbers
serials=("S1600919" "S16900920")

# Loop through each serial number
for serial in "${serials[@]}"; do
  # SSH into remote machine, zip the folder
  ssh "${remote_user}@${remote_host}" "cd ${remote_base_dir} && zip -r ${serial}.zip ${serial}"

  # Ensure local directory exists
  mkdir -p "${local_base_dir}"

  # SCP the zipped file to local directory
  scp "${remote_user}@${remote_host}:${remote_base_dir}/${serial}.zip" "${local_base_dir}/"

  # Optional: Remove the zip file from the remote server after transfer
  ssh "${remote_user}@${remote_host}" "rm ${remote_base_dir}/${serial}.zip"
done
