import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

def extract_scalars_from_event_files(event_files, tag_filter=None):
    """
    Extract scalar values from TensorBoard event files.
    event_files: List of event files to read.
    tag_filter: Optional filter for specific tags (e.g., "loss", "accuracy").
    Returns a dictionary with tag names as keys and lists of (step, value) as values.
    """
    scalars = {}
    
    for event_file in event_files:
        for event in summary_iterator(event_file):
            if event.HasField('summary'):
                for value in event.summary.value:
                    # Print all tags to help identify what's available
                    if tag_filter is None or value.tag == tag_filter:
                        if value.tag not in scalars:
                            scalars[value.tag] = []
                        scalars[value.tag].append((event.step, value.simple_value))
    
    return scalars

# Directory where event files are stored
logdir = "./runs/Dec03_01-16-20_DESKTOP-G2PC6EC"
print('Directory:', logdir)

# List all event files in the directory
event_files = [os.path.join(logdir, f) for f in os.listdir(logdir) if f.startswith("events.out.tfevents.")]
print('Found event files:', event_files)

# Extract scalars (no tag filter, all scalars will be extracted)
scalars = extract_scalars_from_event_files(event_files, tag_filter=None)
print('Extracted scalars:')

# Define output directory to save the plots
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

# Function to sanitize the tag names for valid file names
def sanitize_tag_name(tag):
    # Replace slashes and other invalid characters with underscores
    return tag.replace('/', '_').replace('\\', '_')

# Plot extracted scalar data and save as images
for tag, data in scalars.items():
    steps, values = zip(*data)
    
    # Plotting the scalar values
    plt.scatter(steps, values, label=tag)
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title(f"Scalar Values Over Time - {tag}")
    plt.legend()
    
    # Sanitize the tag to create a valid filename
    sanitized_tag = sanitize_tag_name(tag)
    
    # Save the plot as an image
    plot_filename = os.path.join(output_dir, f"{sanitized_tag}_plot.png")
    plt.savefig(plot_filename)
    print(f"Saved plot for {tag} as {plot_filename}")
    
    # Clear the plot to avoid overlapping with the next plot
    plt.clf()

print("Plots saved successfully.")
