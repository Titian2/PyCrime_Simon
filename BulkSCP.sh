base_dir='/Users/simon/Desktop/CRIME_BACKUP'
remote_dir='PyCrime_machine:/mnt/data/CRIME/results'

# Loop through each serial number
for serial in S1600962; do 
  # Loop through each pattern
  for pattern in 'Grouped_Output_{serial}_*.txt'  'AverageLoss_{serial}_*.png'; do
    # Replace {serial} placeholder with the actual serial number in the pattern
    actual_pattern="${pattern/\{serial\}/$serial}"
    # Use find to search for files matching the pattern and scp them to the remote directory
    find "$base_dir" -name "$actual_pattern" -exec scp {} "$remote_dir/$serial/" \;
  done
done

