import numpy as np
import time as tm

def LJ_pot(position1,position2):
    '''Lennard-Johns potential energy function'''
    return 4*(np.linalg.norm(position1-position2)**(-12)-np.linalg.norm(position1-position2)**(-6))

def Morse_pot1(position1,position2):
    '''Morse potential energy function with parameter r'''
    return (1-np.exp(-np.linalg.norm(position1-position2)+1))**2

def Morse_pot2(position1,position2):
    '''Morse potential energy function with parameter r'''
    return (1-np.exp(-np.linalg.norm(position1-position2)+2))**2

def potential_energy(potential,config):
    '''Function returning the total potential energy of the system'''
    E=0
    for i in range(0,len(config)-1):
        for j in range(i+1,len(config)):
            E+=potential(config[i],config[j])
    return E

def grad(potential,config,delta):
    '''Function returning the gradient of the PES at config'''
    return (potential_energy(potential,config+delta)-potential_energy(potential,config-delta))/(2*np.linalg.norm(delta))

def get_force(potential,config,i,displacement):
    '''Function retuning the force on atom i'''
    forces=np.empty([len(config),3])
    gradient=np.empty(3)
    for j in range(0,3):
        delta=np.zeros([len(config),3])
        delta[i][j]=displacement
        gradient[j]=grad(potential,config,delta)
    forces[i]=gradient
    return -forces

def atom_E(potential,config):
    '''Function returning the sum of pair energies of each atom'''
    N=len(config)
    E=np.zeros(N)
    for i in range(0,N):
        for j in range(0,N):
            if i!=j:
                E[i]+=potential(config[i],config[j])
    return E

def stoch_grad_descent(stepsize, max_step, convergence_cutoff,r,potential):
    '''Stochastic gradient descent local minimiser'''
    t0=tm.time()
    difference=1
    ener0=potential_energy(potential,r)
    numb=0
    N=len(r)
    L=2**(1/6)*N**(1/3)
    v_=np.zeros([N,3])
    while difference>convergence_cutoff and numb<max_step:
        for j in range(0,N):
            forces=get_force(potential,r,j,1e-4)
            if np.linalg.norm(forces)>L/stepsize: #avoid kicking out particles from the cluster if the force is too large on them
                stepsize_small=1/np.linalg.norm(forces)
                r+=0.9*v_+stepsize_small*forces
                v_=stepsize_small*forces
            else:
                r+=0.9*v_+stepsize*forces
                v_=stepsize*forces
        ener1=potential_energy(potential,r)
        difference=np.abs(ener1-ener0)
        ener0=ener1.copy()
        numb+=1
    t1=tm.time()
    print("The algorithm converged in "+str(numb)+" steps and it took "+str(t1-t0)+" seconds")
    print("The energy is: "+str(ener0)) 
    print()
    return ener0,r

def random_hop_min(stepsize, max_step, convergence_cutoff,r,potential,i):
    '''Generates a random radial displacement of atom i, minimises it the energy first with the rest of the atoms frozen'''
    '''Returns the minimum energy and configuration generated this way'''
    config=r.copy()
    COM=np.mean(config,axis=0)
    Rs=[]
    for pos in config:
        Rs.append(np.linalg.norm(COM-pos))
    R=max(Rs)                               #Set R to be the maximum distance from the COM
    theta=np.random.uniform(0,np.pi/2)
    phi=np.random.uniform(0,np.pi)
    x=R*np.sin(theta)*np.cos(phi)
    y=R*np.sin(theta)*np.sin(phi)   
    z=R*np.cos(theta)
    config[i]=COM+np.array([x,y,z])
    ener0=potential_energy(potential,config)
    step_number=0
    difference=1
    while difference>0.01 and step_number<1e3:
        force=get_force(potential,config,i,1e-4)[i]
        config[i]+=5e-3/np.linalg.norm(force)*force
        ener1=potential_energy(potential,config)
        difference=np.abs(ener1-ener0)
        ener0=ener1.copy()
        step_number+=1
    return stoch_grad_descent(stepsize, max_step, convergence_cutoff,config,potential)

def init_random(N,r_e):
    '''initialises the position of N particles randomly with given density parameter r_e'''
    L=r_e*N**(1/3)
    config=np.empty([N,3])
    for i in range(0,N):
        for j in range(0,3):
            config[i][j]=np.random.uniform(0,1)*L
    return config

def write_positions(file_name,r,e):
    '''Creates and .xyz file containing the energy and position of particles'''
    file = open(file_name,"w") 
    file.write(str(len(r))+"\n")
    file.write("The energy is "+str(e)+"\n")
    for i in range(0,len(r)):
        file.write("Ne "+str(r[i][0])+" "+str(r[i][1])+" "+str(r[i][2])+"\n")
    file.close()
    
def read_positions(file_name):
    '''Reads a .xyz file, returns energy and positions of particles'''
    file=open(file_name,"r") 
    line_numb=0
    for line in file:
        if line_numb==0:
            N=int(line.split()[0])
            r=np.empty([N,3])
        elif line_numb==1:
            l=line.split()
            E=float(l[3])
        else:
            l=line.split()
            for j in range(1,4):
                r[line_numb-2][j-1]=float(l[j])
        line_numb+=1
    file.close()
    return E, r
