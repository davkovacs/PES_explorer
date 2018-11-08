LJ_7_minimiser.py and Morse_7_minimiser.py are the two minimisers and functions.py just contains the functions used in both

Description of the cluster minimiser algorythm:

The design of the minimiser was inspired by the paper: J. Phys. Chem. A 1997, 101, 5111-5116 

First the position of N atoms is randomly generated with a set density
Their position is minimised using a modified stochastic gradient algorythm
This local minima is escapeped by a series of basin hopping steps where the atom with the maximum pair energy
is moved randomly on a shell around th centre of masss. It's position is minimised with the position of all other
atoms fixed. Then a full minimisation is carried out and it is assessed whether it improved the minima or not. 
This basin hopping is terminated when the ratio of the minimal and maximal pair energy in the system 
is reaching a threshold.
Finally an other stoch grad descent minimisation is carried out with tighter convergence criteria. 
