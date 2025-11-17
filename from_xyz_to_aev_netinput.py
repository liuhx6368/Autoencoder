import sys,os,ase
from ase import Atoms
import numpy as np
from Atomic_Environment_Vectors import calculate_aev

def ase_atoms_to_aev(atoms: Atoms, species_map: dict = None) -> np.ndarray:
   
    if species_map is None:
        species_map = {'H': 0,'C': 1, 'N': 2, 'O': 3}
    
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    species = []
    for symbol in symbols:
        if symbol not in species_map:
            raise ValueError(f"{symbol} is out of range")
        species.append(species_map[symbol])
    
    aev = calculate_aev(positions, species)
    return aev

def get_Atoms(i):
    with open (i+'.xyz') as f:
        t=f.read().splitlines()[2:]
    a=[]
    for i in range(len(t)):
        t[i]=t[i].split()
        a.append(ase.Atom(t[i][0],[float(t[i][1]),float(t[i][2]),float(t[i][3])],index=i))
    return ase.atoms.Atoms(a)


def get_vector(i):
    aev = ase_atoms_to_aev(get_Atoms(i))
    return aev.flatten()

X=[]
for i in range(1,1001,1):
    X.append(get_vector(str(i)))

np.save('AEV.npy',X) 
