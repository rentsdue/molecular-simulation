from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import mdtraj as md
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ========== Start Timer ==========
start_time = time.time()

# ========== Parameters ==========
cutoff = 0.4
threshold = 0.5
progress_steps = 100
num_clusters = 4
max_residue_distance = 10

# ========== Paths ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")

# ========== Load Trajectories ==========
traj_bound = md.load(os.path.join(src_dir, "md_skip_gr_ligand.xtc"),
                     top=os.path.join(src_dir, "gr_ligand.pdb"))
traj_unbound = md.load(os.path.join(src_dir, "md_skip_gr_only.xtc"),
                       top=os.path.join(src_dir, "gr_only.pdb"))

# ========== Chain Selection ==========
protein_bound = traj_bound.atom_slice(traj_bound.topology.select("protein and chainid 1"))
protein_unbound = traj_unbound.atom_slice(traj_unbound.topology.select("protein and chainid 0"))

# ========== Residue Matching ==========
def get_residue_dict(topology):
    return {(res.chain.index, res.resSeq): res for res in topology.residues}

res_dict_bound = get_residue_dict(protein_bound.topology)
res_dict_unbound = get_residue_dict(protein_unbound.topology)
matched_keys = sorted(set(res_dict_bound.keys()) & set(res_dict_unbound.keys()))
residues_bound = [res_dict_bound[k] for k in matched_keys]
residues_unbound = [res_dict_unbound[k] for k in matched_keys]
residue_labels = [f"{res.name}{res.resSeq}_chain{res.chain.index}" for res in residues_bound]
residue_pairs = list(itertools.combinations(range(len(matched_keys)), 2))

# ========== Contact Occupancy ==========
def contact_occupancy_fast(traj, residues, pairs, cutoff_nm, label):
    n_frames = traj.n_frames
    occupancies = {}
    progress_intervals = max(1, len(pairs) // progress_steps)
    for idx, (i, j) in enumerate(pairs):
        if idx % progress_intervals == 0:
            print(f"[{label}] Progress: {idx * 100 // len(pairs)}%")
        atoms1 = [a.index for a in residues[i].atoms if a.element.symbol != "H"]
        atoms2 = [a.index for a in residues[j].atoms if a.element.symbol != "H"]
        if not atoms1 or not atoms2:
            continue
        atom_pairs = np.array(list(itertools.product(atoms1, atoms2)))
        distances = md.compute_distances(traj, atom_pairs)
        contact_frames = np.any(distances < cutoff_nm, axis=1)
        occupancy = np.sum(contact_frames) / n_frames
        if occupancy > 0:
            occupancies[(i, j)] = occupancy
    return occupancies

occ_bound = contact_occupancy_fast(protein_bound, residues_bound, residue_pairs, cutoff, "bound")
occ_unbound = contact_occupancy_fast(protein_unbound, residues_unbound, residue_pairs, cutoff, "unbound")

# ========== Filter and Build Difference Matrix ==========
n_res = len(matched_keys)
diff_matrix = np.zeros((n_res, n_res))
for (i, j) in residue_pairs:
    delta = occ_unbound.get((i, j), 0) - occ_bound.get((i, j), 0)
    diff_matrix[i, j] = delta
    diff_matrix[j, i] = delta

# ========== Select High-Difference Residues ==========
residue_scores = np.sum(np.abs(diff_matrix), axis=1)
significant_indices = np.where(residue_scores > np.percentile(residue_scores, 75))[0]
diff_submatrix = diff_matrix[significant_indices][:, significant_indices]
labels_sub = [residue_labels[i] for i in significant_indices]

# ========== Dimensionality Reduction & Clustering ==========
pca = PCA(n_components=2)
X_pca = pca.fit_transform(diff_submatrix)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(X_pca)

# ========== Heatmap ==========
plt.figure(figsize=(10, 8))
sns.heatmap(diff_submatrix, cmap="bwr", center=0, square=True,
            xticklabels=False, yticklabels=False, cbar=False)
plt.title("Residue Contact Difference Clusters (PCA + KMeans)")
plt.tight_layout()
plt.savefig("heatmap_diff_clusters_only.png", dpi=300)
plt.close()

# ========== Filter Close High-Difference Pairs ==========
close_pairs = []
for i in range(len(significant_indices)):
    for j in range(i + 1, len(significant_indices)):
        idx1 = significant_indices[i]
        idx2 = significant_indices[j]
        delta = abs(diff_matrix[idx1, idx2])
        if delta > threshold and abs(idx1 - idx2) <= max_residue_distance:
            close_pairs.append((idx1, idx2, delta))

with open("significant_close_pairs.txt", "w") as f:
    f.write("Index1\tIndex2\tDelta\tLabel1\tLabel2\n")
    for i, j, delta in sorted(close_pairs, key=lambda x: -x[2]):
        f.write(f"{i}\t{j}\t{delta:.3f}\t{residue_labels[i]}\t{residue_labels[j]}\n")

# ========== End Timer ==========
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
