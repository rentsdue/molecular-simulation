import mdtraj as md
import os
import itertools
import numpy as np
import time

# ========== Start Timer ==========
start_time = time.time()

# ========== Parameters ==========
cutoff = 0.4    # nm for contact
threshold = 0.5 # occupancy threshold for filtering
progress_steps = 100  # % steps to show

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

# ========== Structural Debug Check ==========
print("=== Structural Debug Check ===")
print(f"Selected from gr_ligand (chain B): {protein_bound.n_atoms} atoms, {len(list(protein_bound.topology.residues))} residues")
print(f"Selected from gr_only (chain A): {protein_unbound.n_atoms} atoms, {len(list(protein_unbound.topology.residues))} residues")
print()

# ========== Residue Matching ==========
def get_residue_dict(topology):
    res_dict = {}
    for res in topology.residues:
        chain_id = res.chain.index
        res_id = res.resSeq
        res_dict[(chain_id, res_id)] = res
    return res_dict

res_dict_bound = get_residue_dict(protein_bound.topology)
res_dict_unbound = get_residue_dict(protein_unbound.topology)
matched_keys = sorted(set(res_dict_bound.keys()) & set(res_dict_unbound.keys()))

residues_bound = [res_dict_bound[k] for k in matched_keys]
residues_unbound = [res_dict_unbound[k] for k in matched_keys]
residue_pairs = list(itertools.combinations(range(len(matched_keys)), 2))

print(f"Matched residues: {len(matched_keys)}")
print(f"Total residue pairs: {len(residue_pairs)}")

with open("matched_residues.txt", "w") as f:
    for idx, key in enumerate(matched_keys):
        f.write(f"{idx}: Chain {key[0]}, ResSeq {key[1]}\n")

# ========== Optimized Contact Occupancy Function ==========
def contact_occupancy_fast(traj, residues, pairs, cutoff_nm, label):
    print(f"\nComputing contact occupancy ({label}) [fast]...")
    n_frames = traj.n_frames
    occupancies = {}

    progress_intervals = max(1, len(pairs) // progress_steps)
    for idx, (i, j) in enumerate(pairs):
        if idx % progress_intervals == 0:
            print(f"Progress: {idx * 100 // len(pairs)}%")

        res1 = residues[i]
        res2 = residues[j]

        atoms1 = [a.index for a in res1.atoms if a.element.symbol != "H"]
        atoms2 = [a.index for a in res2.atoms if a.element.symbol != "H"]

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

# ========== Compute Occupancies ==========
occ_bound = contact_occupancy_fast(protein_bound, residues_bound, residue_pairs, cutoff, "bound")
occ_unbound = contact_occupancy_fast(protein_unbound, residues_unbound, residue_pairs, cutoff, "unbound")

# ========== Filter by Occupancy Threshold ==========
filtered_pairs = [
    pair for pair in residue_pairs
    if occ_bound.get(pair, 0) > threshold or occ_unbound.get(pair, 0) > threshold
]
print(f"Filtered to {len(filtered_pairs)} pairs with occupancy > {threshold}")

# ========== Generate PyMOL Script (Excluding Close Residue Contacts) ==========
pml_file = "contacts_longrange.pml"
structure_path = os.path.join(src_dir, "gr_ligand.pdb")  # reference structure

with open(pml_file, "w") as f:
    f.write(f"load {structure_path}, structure\n")
    f.write("hide everything\nshow cartoon, structure\n\n")

    n_written = 0
    for i, j in filtered_pairs:
        resi1 = residues_bound[i].resSeq
        resi2 = residues_bound[j].resSeq

        if abs(resi1 - resi2) <= 20:
            continue  # Skip close-range contacts

        name = f"contact_{resi1}_{resi2}"
        f.write(
            f"distance {name}, "
            f"structure and resi {resi1} and name CA, "
            f"structure and resi {resi2} and name CA\n"
        )
        n_written += 1

    f.write("\nhide labels\n")
    f.write("color green, name contact_*\n")
    f.write("set dash_color, cyan\n")
    f.write("set dash_width, 2\n")
    f.write("zoom\n")

print(f"PyMOL script written to: {pml_file} (long-range only, {n_written} contacts)")

# ========== End Timer ==========
end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
