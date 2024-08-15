from Bio.PDB import *
from default_config.chemistry import radii, polarHydrogens

"""
xyzrn.py: Read a pdb file and output it is in xyzrn for use in MSMS
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

# TODO: need to include ligands here for full msms surface?
def output_pdb_as_xyzrn(pdbfilename, xyzrnfilename):
    """
        pdbfilename: input pdb filename
        xyzrnfilename: output in xyzrn format.
    """
    parser = PDBParser()
    struct = parser.get_structure(pdbfilename, pdbfilename)
    outfile = open(xyzrnfilename, "w")
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        reskey = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        atomtype = name[0]

        color = "Green"
        coords = None
        if atomtype in radii and resname in polarHydrogens:
            if atomtype == "O":
                color = "Red"
            if atomtype == "N":
                color = "Blue"
            if atomtype == "H":
                if name in polarHydrogens[resname]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, color
            )
        if coords is not None:
            outfile.write(coords + " " + radii[atomtype] + " 1 " + full_id + "\n")

def output_pdb_as_xyzrn2(pdbfilename, xyzrnfilename, pout=False):
    """
        pdbfilename: input pdb filename
        xyzrnfilename: output in xyzrn format.
    """
    parser = PDBParser()
    struct = parser.get_structure(pdbfilename, pdbfilename)
    outfile = open(xyzrnfilename, "w")
    for atom in struct.get_atoms():
        name = atom.get_name()
        residue = atom.get_parent()
        # Ignore hetatms.
        # if residue.get_id()[0] != " ":
        #     continue
        resname = residue.get_resname()
        reskey = residue.get_id()[1]
        chain = residue.get_parent().get_id()
        atomtype = name[0]
        if resname == "Fe2":
            atomtype = "Fe2p"

        color = "Green"
        coords = None
        if atomtype in radii and resname in polarHydrogens:
            if atomtype == "O":
                color = "Red"
            if atomtype == "N":
                color = "Blue"
            if atomtype == "H":
                if name in polarHydrogens[resname]:
                    color = "Blue"  # Polar hydrogens
            coords = "{:.06f} {:.06f} {:.06f}".format(
                atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            )
            insertion = "x"
            if residue.get_id()[2] != " ":
                insertion = residue.get_id()[2]
            full_id = "{}_{:d}_{}_{}_{}_{}".format(
                chain, residue.get_id()[1], insertion, resname, name, color
            )
        else:
            # print(atom, atomtype, resname)
            pass
        if coords is not None:
            if pout:
                print("outputting xyzrnfilename", outfile)
            outfile.write(coords + " " + radii[atomtype] + " 1 " + full_id + "\n")
