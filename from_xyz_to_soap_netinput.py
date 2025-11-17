import numpy as np
import ase,sys,os
from dscribe.descriptors import SOAP

ase_atoms_to_soap = SOAP(species = ["C", "N", "H"],
                         periodic = False,
                         r_cut = 4.0,
                         n_max = 6,
                         l_max = 4,
                         rbf = 'polynomial'
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
    soap=ase_atoms_to_soap.create(a)
    return soap.flatten()

X=[]
for i in range(1,1001,1):
    X.append(get_vector(get_Atoms(str(i))))

np.save('SOAP.npy',X)

