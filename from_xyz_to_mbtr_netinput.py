import numpy as np
import ase,sys,os
from dscribe.descriptors import MBTR

ase_atoms_to_mbtr = MBTR(species=["C","N","O"],
                         geometry={"function": "cosine"},
                         #k=1:atomic_number;k=2:distance,inverse_distance;k=3:angle,cosine
                         grid={"min":-1.0, "max":1.0, "n":50, "sigma": 0.1},
                         weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
                         #k=1:unity;k=2:unity,exp,inverse_square;k=3:unity,exp,smooth_cutoff
                         normalize_gaussians=False,
                         periodic=False,
                         normalization="l2",
                         #none;l2;n_atoms;valle_oganov
                         #sparse=True,
                         dtype="float64"
                        )

def get_Atoms(i):
    with open (i+'.xyz') as f:
        t=f.read().splitlines()[2:]
    a=[]
    for i in range(len(t)):
        t[i]=t[i].split()
        a.append(ase.Atom(t[i][0],[float(t[i][1]),float(t[i][2]),float(t[i][3])],index=i))
    return ase.atoms.Atoms(a)

def get_vector(a):
    mbtr=ase_atoms_to_mbtr.create(a)
    return mbtr

X=[]
for i in range(1,1001,1):
    X.append(get_vector(get_Atoms(str(i))))

np.save('MBTR.npy',X)

