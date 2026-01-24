from ultralytics.utils.plotting import plot_results
import shutil
import os

# Define the path to your results.csv file
csv_path = '/home/choneil/repos/ultralytics/runs/detect/train28/results.csv' 

# (Optional) If in a restricted environment like Kaggle, copy the file to a writable directory
# dst_path = "/kaggle/working/"
# shutil.copy(src_path, dst_path)
# plot_results(os.path.join(dst_path, 'results.csv'))

# Plot the results
plot_results(csv_path)

