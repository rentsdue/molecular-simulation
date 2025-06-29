import mdtraj as md
import os

# ========== Paths ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "..", "src")

ligand_pdb = os.path.join(src_dir, "gr_ligand.pdb")
unbound_pdb = os.path.join(src_dir, "gr_only.pdb")
ligand_xtc = os.path.join(src_dir, "md_skip_gr_ligand.xtc")
unbound_xtc = os.path.join(src_dir, "md_skip_gr_only.xtc")

# ========== Load topologies ==========
ligand_top = md.load_topology(ligand_pdb)
unbound_top = md.load_topology(unbound_pdb)

print("== Chains in gr_ligand.pdb ==")
for chain in ligand_top.chains:
    print(f"Chain index: {chain.index}, Residues: {len(list(chain.residues))}")

print("\n== Chains in gr_only.pdb ==")
for chain in unbound_top.chains:
    print(f"Chain index: {chain.index}, Residues: {len(list(chain.residues))}")

# ========== Load one frame to test selection ==========
ligand_test = md.load_xtc(ligand_xtc, top=ligand_pdb, frame=0)
unbound_test = md.load_xtc(unbound_xtc, top=unbound_pdb, frame=0)

# ========== Test selection by chain index ==========
ligand_protein_chainB = ligand_test.atom_slice(
    ligand_test.topology.select("protein and chainid 1"))

unbound_protein_chainA = unbound_test.atom_slice(
    unbound_test.topology.select("protein and chainid 0"))

print("\n== Selection Summary ==")
print(f"gr_ligand (chainid 1): {ligand_protein_chainB.n_atoms} atoms, "
      f"{len(list(ligand_protein_chainB.topology.residues))} residues")
print(f"gr_only (chainid 0): {unbound_protein_chainA.n_atoms} atoms, "
      f"{len(list(unbound_protein_chainA.topology.residues))} residues")
