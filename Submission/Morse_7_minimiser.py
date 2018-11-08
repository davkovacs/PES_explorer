import numpy as np
import time as tm
from functions import *

parameter=int(input("What is the Morse-parameter? (1 or 2) "))
assert parameter==1 or parameter==2, "Please type either 1 or 2"

if parameter==1:
    potential=Morse_pot1
    alpha=2.5
elif parameter==2:
    potential=Morse_pot2
    alpha=5.5

N=7
min_e=N*(N-1)/2

configuration=init_random(N,0.9)
min_e,min_r=stoch_grad_descent(1e-3,5e4,5e-5,configuration,potential)

#basin hopping
atom_energies=atom_E(potential,min_r)
found_less=0
while max(atom_energies)<min(atom_energies)*alpha:
    max_ener_atom=np.argmax(atom_energies)
    e,r=random_hop_min(1e-3,1e4,2e-4,min_r,potential,max_ener_atom)
    if e<min_e:
        min_e=e.copy()
        min_r=r.copy()
        config=r.copy()
        atom_energies=atom_E(potential,min_r)

min_e,min_r=stoch_grad_descent(1e-5,2e5,1e-6,min_r,potential)
write_positions("Morse"+str(parameter)+"_cluster_positions_"+str(N)+"_opt.xyz",min_r,min_e)
