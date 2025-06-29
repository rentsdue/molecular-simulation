import numpy as np
import matplotlib.pyplot as plt

# Load data
import os

# Get directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")

files = [
    "rmsf_gr_only_1.xvg",
    "rmsf_gr_only_2.xvg",
    "rmsf_gr_only_3.xvg"
]

all_rmsf = []

for fname in files:
    x = []
    y = []
    file_path = os.path.join(src_dir, fname)
    with open(file_path) as file:
        for line in file:
            if line.startswith(("#", "@")):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
    all_rmsf.append(y)

rmsf_mean = np.mean(all_rmsf, axis=0)
rmsf_std = np.std(all_rmsf, axis=0)
x_vals = x

plt.figure(figsize=(12, 6))
plt.plot(x_vals, rmsf_mean, color="blue", label="Mean RMSF")
plt.fill_between(
    x_vals, rmsf_mean - rmsf_std, rmsf_mean + rmsf_std,
    color="gray", alpha=0.3, label="±1 SD"
)

# # helices = [
# #     (32, 51), (92, 94), (140, 168), (170, 172), (175, 205),
# #     (207, 215), (220, 222), (224, 256), (262, 291), (304, 334),
# #     (345, 360), (362, 368), (372, 374), (378, 402), (407, 422)
# # ]

# # Highlight helix regions
# for (start, end) in helices:
#     plt.axvspan(start, end, color="peachpuff", alpha=0.3)

# # Prepare xticks
# midpoints = [(s + e) / 2 for s, e in helices]
# xtick_labels = [f"{s}–{e}" for s, e in helices]
# plt.xticks(midpoints, xtick_labels, rotation=90)

# Add H1, H2 labels at the top
# y_top = (rmsf_mean + rmsf_std).max() * 1.1
# for i, mid in enumerate(midpoints, start=1):
#     plt.text(mid, y_top, f"H{i}", ha="center", va="bottom", fontsize=8, color="black")

# Dummy legend
plt.scatter([], [], color="peachpuff", alpha=0.3, label="Helices")

plt.xlabel("Residue Range", labelpad=20)  # Shift x-label down
plt.ylabel("RMSF (nm)")
plt.title("Mean RMSF with Helices Annotated at Top", pad=35)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
