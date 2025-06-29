import mdtraj as md
import os
import itertools
import numpy as np
import scipy.linalg
import seaborn as sns
import matplotlib.pyplot as plt
import time

# ========== Start Timer ==========
start_time = time.time()

# ========== Parameters ==========
cutoff = 0.4    # nm for contact
threshold = 0.5 # occupancy threshold
progress_steps = 100  # 1% resolution

# ========== Paths ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")

# ========== Load Trajectories ==========
print("Loading trajectories...")
traj_bound = md.load(os.path.join(src_dir, "md_skip_gr_ligand.xtc"),
                     top=os.path.join(src_dir, "gr_ligand.pdb"))
traj_unbound = md.load(os.path.join(src_dir, "md_skip_gr_only.xtc"),
                       top=os.path.join(src_dir, "gr_only.pdb"))

# ========== Chain Selection ==========
protein_bound = traj_bound.atom_slice(
    traj_bound.topology.select("protein and chainid 1"))  # Chain B
protein_unbound = traj_unbound.atom_slice(
    traj_unbound.topology.select("protein and chainid 0"))  # Chain A

# ========== Residue Matching ==========
def get_residue_dict(topology):
    return { (res.chain.index, res.resSeq): res for res in topology.residues }

res_dict_bound = get_residue_dict(protein_bound.topology)
res_dict_unbound = get_residue_dict(protein_unbound.topology)
matched_keys = sorted(set(res_dict_bound) & set(res_dict_unbound))

residues_bound = [res_dict_bound[k] for k in matched_keys]
residues_unbound = [res_dict_unbound[k] for k in matched_keys]
residue_pairs = list(itertools.combinations(range(len(matched_keys)), 2))
n_res = len(matched_keys)

print(f"Matched residues: {n_res}")
print(f"Total residue pairs: {len(residue_pairs)}")

# ========== Contact Occupancy Function ==========
def contact_occupancy_fast(traj, residues, pairs, cutoff_nm, label):
    print(f"\nComputing contact occupancy ({label})...")
    n_frames = traj.n_frames
    occupancies = {}
    step = max(1, len(pairs) // progress_steps)

    for idx, (i, j) in enumerate(pairs):
        if idx % step == 0:
            print(f"Progress: {idx * 100 // len(pairs)}%")

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

    print("Progress: 100% (done)")
    return occupancies

# ========== Compute & Process Communicability ==========
def process_system(name, residues, traj, residue_pairs):
    occ = contact_occupancy_fast(traj, residues, residue_pairs, cutoff, name)

    # Filter by threshold
    filtered = [pair for pair in residue_pairs if occ.get(pair, 0) > threshold]
    print(f"{name}: {len(filtered)} pairs above threshold {threshold}")

    # Build adjacency matrix
    print(f"\nBuilding adjacency matrix ({name})...")
    adj_matrix = np.zeros((n_res, n_res))
    step = max(1, len(filtered) // progress_steps)

    for idx, (i, j) in enumerate(filtered):
        if idx % step == 0:
            print(f"Progress: {idx * 100 // len(filtered)}%")
        w = occ.get((i, j), 0)
        adj_matrix[i, j] = w
        adj_matrix[j, i] = w

    print("Progress: 100% (done)")

    # Communicability matrix
    print(f"\nComputing communicability matrix ({name})...")
    comm_matrix = scipy.linalg.expm(adj_matrix)

    # Save results
    csv_path = f"communicability_{name}.csv"
    png_path = f"communicability_{name}.png"
    np.savetxt(csv_path, comm_matrix, delimiter=",", fmt="%.6f")
    print(f"{name} matrix saved to: {csv_path}")

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(comm_matrix, cmap="viridis", square=True, cbar_kws={"label": "Communicability"})
    plt.title(f"{name.capitalize()} Residue Communicability Matrix")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"{name} heatmap saved to: {png_path}")

# Run for both systems
process_system("bound", residues_bound, protein_bound, residue_pairs)
process_system("unbound", residues_unbound, protein_unbound, residue_pairs)

# ========== End Timer ==========
end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
