import numpy as np
import matplotlib.pyplot as plt
import os

def load_rmsf_data(filenames, base_dir=None):
    """Load RMSF data from a list of .xvg files. Assumes all files share the same x-axis."""
    all_y = []
    x_ref = None
    for i, fname in enumerate(filenames):
        x = []
        y = []
        path = os.path.join(base_dir, fname) if base_dir else fname
        with open(path) as file:
            for line in file:
                if line.startswith(("#", "@")):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
        if i == 0:
            x_ref = x
        all_y.append(y)
    return x_ref, np.array(all_y)

# Define input
files = ["rmsf_gr_1.xvg", "rmsf_gr_2.xvg", "rmsf_gr_3.xvg"]
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")  # adjust as needed

# Load data
x_vals, all_rmsf = load_rmsf_data(files, base_dir=src_dir)
rmsf_mean = np.mean(all_rmsf, axis=0)
rmsf_std = np.std(all_rmsf, axis=0)

# Define helices
helices = [
    (32, 51), (92, 94), (140, 168), (170, 172), (175, 205),
    (207, 215), (220, 222), (224, 256), (262, 291), (304, 334),
    (345, 360), (362, 368), (372, 374), (378, 402), (407, 422)
]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(x_vals, rmsf_mean, color="blue", label="Mean RMSF")
plt.fill_between(x_vals, rmsf_mean - rmsf_std, rmsf_mean + rmsf_std,
                 color="gray", alpha=0.3, label="±1 SD")

# Highlight helices
for (start, end) in helices:
    plt.axvspan(start, end, color="peachpuff", alpha=0.3)

# Add xtick labels at helix midpoints
midpoints = [(s + e) / 2 for s, e in helices]
xtick_labels = [f"{s}–{e}" for s, e in helices]
plt.xticks(midpoints, xtick_labels, rotation=90)

# Annotate helix numbers at top
y_top = (rmsf_mean + rmsf_std).max() * 1.1
for i, mid in enumerate(midpoints, start=1):
    plt.text(mid, y_top, f"H{i}", ha="center", va="bottom", fontsize=8)

# Dummy legend entry for helix span
plt.scatter([], [], color="peachpuff", alpha=0.3, label="Helices")

# Labels and formatting
plt.xlabel("Residue Range", labelpad=20)
plt.ylabel("RMSF (nm)")
plt.title("Mean RMSF with Helices Annotated at Top", pad=35)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

