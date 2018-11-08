import numpy as np
import time as tm
from functions import *

min_e=0
N=7

configuration=init_random(N,1.0)
min_e,min_r=stoch_grad_descent(1e-3,5e4,5e-5,configuration,LJ_pot)

#basin hopping
atom_energies=atom_E(LJ_pot,min_r)
found_less=0
while max(atom_energies)>min(atom_energies)*0.65:
    max_ener_atom=np.argmax(atom_energies)
    e,r=random_hop_min(1e-3,1e4,2e-4,min_r,LJ_pot,max_ener_atom)
    if e<min_e:
        min_e=e.copy()
        min_r=r.copy()
        config=r.copy()
        atom_energies=atom_E(LJ_pot,min_r)        

min_e,min_r=stoch_grad_descent(1e-4,1e5,5e-6,min_r,LJ_pot)
write_positions("LJ_cluster_positions_"+str(N)+"_opt.xyz",min_r,min_e) 
