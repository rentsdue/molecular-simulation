import mdtraj as md
import os

# --- Step 1: File setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")
bound_path = os.path.join(src_dir, "gr_ligand.pdb")
unbound_path = os.path.join(src_dir, "gr_only.pdb")

# --- Step 2: Load structures ---
print("Loading structures...")
bound = md.load(bound_path)
unbound = md.load(unbound_path)

# --- Step 3: Define loop residue ranges ---
loop_ranges = [(29, 31), (52, 77), (85, 91)]  # 1-based residue numbers

# --- Step 4: Select heavy atoms from those loops ---
top = bound.topology
selected_atoms = []

for start, end in loop_ranges:
    for atom in top.atoms:
        res_id = atom.residue.index + 1  # Convert to 1-based indexing
        if start <= res_id <= end and atom.element.symbol != 'H':
            selected_atoms.append(atom.index)

print(f"Selected {len(selected_atoms)} heavy atoms in loop regions.")

# --- Step 5: Generate unique atom pairs within loops ---
pairs = [(i, j) for i in selected_atoms for j in selected_atoms if i < j]
print(f"Generated {len(pairs)} atom pairs for contact analysis.")

# --- Step 6: Compute pairwise distances ---
cutoff = 0.4  # in nanometers
print(f"Computing contacts using {cutoff} nm cutoff...")

dist_bound = md.compute_distances(bound, pairs)[0]
dist_unbound = md.compute_distances(unbound, pairs)[0]

contacts_bound = dist_bound < cutoff
contacts_unbound = dist_unbound < cutoff

# --- Step 7: Identify formed and broken contacts ---
formed_pairs = [pairs[i] for i in range(len(pairs)) if contacts_unbound[i] and not contacts_bound[i]]
broken_pairs = [pairs[i] for i in range(len(pairs)) if contacts_bound[i] and not contacts_unbound[i]]

# --- Step 8: Reporting ---
def describe_atom_pair(pair, topology):
    a, b = pair
    atom_a = topology.atom(a)
    atom_b = topology.atom(b)
    return f"{atom_a} — {atom_b}"

print("\nFormed Contacts (unbound only):")
for pair in formed_pairs:
    print("  Formed:", describe_atom_pair(pair, top))

print("\nBroken Contacts (bound only):")
for pair in broken_pairs:
    print("  Broken:", describe_atom_pair(pair, top))
