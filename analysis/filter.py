from sklearn.decomposition import PCA
import mdtraj as md
import os
import itertools
import numpy as np
import time

# ========== Start Timer ==========
start_time = time.time()

# ========== Parameters ==========
cutoff = 0.4  # nm
threshold = 0.5  # contact difference threshold
min_residue_separation = 11  # must be more than 10 residues apart

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
def contact_occupancy_fast(traj, residues, pairs, cutoff_nm):
    n_frames = traj.n_frames
    occupancies = {}
    total = len(pairs)
    progress_interval = max(1, total // 100)  # Every 1%

    for idx, (i, j) in enumerate(pairs):
        if idx % progress_interval == 0:
            print(f"Progress: {idx * 100 // total}%")

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

occ_bound = contact_occupancy_fast(protein_bound, residues_bound, residue_pairs, cutoff)
occ_unbound = contact_occupancy_fast(protein_unbound, residues_unbound, residue_pairs, cutoff)

# ========== Identify Significant Long-Range Contact Changes ==========
significant_long_range = []
for (i, j) in residue_pairs:
    if abs(i - j) >= min_residue_separation:
        occ_b = occ_bound.get((i, j), 0.0)
        occ_u = occ_unbound.get((i, j), 0.0)
        delta = abs(occ_u - occ_b)
        if delta > threshold:
            significant_long_range.append((i, j, delta))

# ========== Output ==========
output_path = "significant_long_range_pairs.txt"
with open(output_path, "w") as f:
    f.write("Index1\tIndex2\tDelta\tLabel1\tLabel2\n")
    for i, j, delta in sorted(significant_long_range, key=lambda x: -x[2]):
        f.write(f"{i}\t{j}\t{delta:.3f}\t{residue_labels[i]}\t{residue_labels[j]}\n")

# ========== End Timer ==========
end_time = time.time()
print(f"Finished in {end_time - start_time:.2f} seconds. Output saved to {output_path}")
