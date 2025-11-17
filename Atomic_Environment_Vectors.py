import numpy as np
from typing import List, Tuple

Rc_radial = 3.0
Rc_angular = 8.0
eta_r = np.array([8.0, 4.0, 2.0])
Rs_r = np.array([0.3, 0.9, 1.5, 2.1, 2.7])
eta_a = np.array([16.0, 8.0, 4.0])
zeta = np.array([1.0, 2.0, 4.0, 8.0])
Rs_a = np.array([0.9, 1.5, 2.1, 2.7])
theta_s = np.array([0.0, 1.047, 2.094, 3.142])

def cutoff_function(R: np.ndarray, Rc: float) -> np.ndarray:
    result = np.zeros_like(R)
    mask = R <= Rc
    result[mask] = 0.5 * (np.cos(np.pi * R[mask] / Rc) + 1)
    return result

def calculate_radial_terms(distances: np.ndarray, species: List[int]) -> np.ndarray:
   
    n_atoms = len(species)
    n_species = 4  # H, C, N, O
    n_eta_r = len(eta_r)
    n_Rs_r = len(Rs_r)
    
  
    radial_aev = np.zeros((n_atoms, n_species * n_eta_r * n_Rs_r))
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
                
            R_ij = distances[i, j]
            if R_ij > Rc_radial:
                continue
                
            j_species = species[j]
            if j_species >= n_species: 
                continue
                
            
            f_c = cutoff_function(R_ij, Rc_radial)
            for e_idx, e in enumerate(eta_r):
                for r_idx, r in enumerate(Rs_r):
                    index = j_species * n_eta_r * n_Rs_r + e_idx * n_Rs_r + r_idx
                    radial_aev[i, index] += np.exp(-e * (R_ij - r)**2) * f_c
    
    return radial_aev

def calculate_angular_terms(distances: np.ndarray, positions: np.ndarray, species: List[int]) -> np.ndarray:
    
    n_atoms = len(species)
    n_species_pairs = 10  # H-H, H-C, H-N, H-O, C-C, C-N, C-O, N-N, N-O, O-O
    n_eta_a = len(eta_a)
    n_Rs_a = len(Rs_a)
    n_zeta = len(zeta)
    n_theta_s = len(theta_s)
    
   
    angular_aev = np.zeros((n_atoms, n_species_pairs * n_eta_a * n_Rs_a * n_zeta * n_theta_s))
    
   
    species_pairs = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    pair_to_index = {(min(a, b), max(a, b)): idx for idx, (a, b) in enumerate(species_pairs)}
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
                
            R_ij = distances[i, j]
            if R_ij > Rc_angular:
                continue
                
            j_species = species[j]
            
            for k in range(j+1, n_atoms):
                if i == k:
                    continue
                    
                R_ik = distances[i, k]
                if R_ik > Rc_angular:
                    continue
                    
                k_species = species[k]
                
                
                pair = (min(j_species, k_species), max(j_species, k_species))
                if pair not in pair_to_index:
                    continue
                    
                pair_idx = pair_to_index[pair]
                
               
                r_ij = positions[j] - positions[i]
                r_ik = positions[k] - positions[i]
                cos_theta = np.dot(r_ij, r_ik) / (R_ij * R_ik + 1e-10)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                
             
                f_c_ij = cutoff_function(R_ij, Rc_angular)
                f_c_ik = cutoff_function(R_ik, Rc_angular)
                
                for e_idx, e in enumerate(eta_a):
                    for r_idx, r in enumerate(Rs_a):
                        for z_idx, z in enumerate(zeta):
                            for t_idx, t in enumerate(theta_s):
                             
                                factor = 2**(1-z) * (1 + cos_theta - np.cos(t))**z
                                value = np.exp(-e * ((R_ij + R_ik)/2 - r)**2) * f_c_ij * f_c_ik * factor
                                
                              
                                index = (pair_idx * n_eta_a * n_Rs_a * n_zeta * n_theta_s + 
                                        e_idx * n_Rs_a * n_zeta * n_theta_s + 
                                        r_idx * n_zeta * n_theta_s + 
                                        z_idx * n_theta_s + 
                                        t_idx)
                                
                                angular_aev[i, index] += value
    
    return angular_aev

def calculate_aev(positions: np.ndarray, species: List[int]) -> np.ndarray:
    
    n_atoms = len(positions)
    
    
    distances = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
   
    radial = calculate_radial_terms(distances, species)
    angular = calculate_angular_terms(distances, positions, species)
    
    
    aev = np.hstack([radial, angular])
    return aev
