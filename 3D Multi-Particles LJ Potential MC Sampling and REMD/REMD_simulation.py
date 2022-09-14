### Replica Exchange Molecular Dynamics ###
from Monte_Carlo_Simulation import init_coor


### Define functions ###

def decide_replica(T2,V2,x2,y2,z2,T1,V1,x1,y1,z1,counter):   
  '''
  This function is used to decide accepting/rejecting a replica exchange
  Input: Temp, Potential, coordinates of two states
  If accept the exchange: return the switched coordinates and potential
  If reject the exchange: return the original coordinates and potential
  !!!Problem: I'm not sure if this decide process is correct!!!
  '''
  beta2 = 1/(kb*T2)
  beta1 = 1/(kb*T1)
  ratio = np.exp((beta2-beta1)*(V2-V1))
  if ratio >= 1:
    counter += 1
    return V1,x1,y1,z1,V2,x2,y2,z2,counter   # accpet the exchange
  else:
    return V2,x2,y2,z2,V1,x1,y1,z1,counter   # reject the exchange



def replica_ex(N,Nstep,Nomve,T,num_ex):      # num_ex is how many replica exchanges to do 
  '''
  This function is used to apply the replica exchange from the very start
  Step1: initialize N data points
  Step2: Do Monte Carlo to the initialized data points for Nsteps
  Step3: Apply replica_exchange in a loop (eg. for a list of temp [T0,T1,T2,T3,T4], do exchange between T0/T1 then do T2/T3, then do T4&T0, then do T1/T2 ... )
  '''
  ### Step1: initialize N data points ###
  length = len(T)                           # T is a list that has odd number of elements, save the length of T list
  x_ini,y_ini,z_ini,V_ini = init_coor(N)    # generate initial coordinates and potential (default N=50, boxsize:20*20*0, stepsize:dx=dy=2)
  x = [[] for i in range(length)]           # x is a 2d-array. x[i][j] means the x_coord of j'th point(N) at T[i]
  y = [[] for i in range(length)]
  z = [[] for i in range(length)]
  V = [0 for i in range(length)]            # V is a list. V[i] is the potential at T[i]
  V_2d = [[] for i in range(length)]        # V_2d is a 2d-array. V_2d[i][j] is the total LJ potential at T[i] with Nstep*j steps

  ### Step2: Do MC to initialized data points at different Temperatures           
  for i in range(len(T)): 
    x[i], y[i], z[i], V[i], acc_rate = MC_coor(N,Nstep,Nmove,T[i],x_ini,y_ini,z_ini,V_ini)
    V_2d[i].append(V[i])
  
  ### Step3: Apply replica_exchange in a loop ###
  counter = 0
  for i in range(num_ex):
    # pick key1 and key2 (which two temp for replica exchange)
    key1 = (2*i)%length                              
    key2 = (2*i+1)%length
    # decide if taking the replica exchange for key1 and key2 temperatures
    V[key1],x[key1],y[key1],z[key1],V[key2],x[key2],y[key2],z[key2],counter = decide_replica(T[key1],V[key1],x[key1],y[key1],z[key1],T[key2],V[key2],x[key2],y[key2],z[key2],counter)

    # After replica exchange, do MC for Nstep at key1 and key2 temperatures
    x[key1],y[key1],z[key1],V[key1], acc_rate = MC_coor(N,Nstep,Nmove,T[key1],x[key1],y[key1],z[key1],V[key1])
    x[key2],y[key2],z[key2],V[key2], acc_rate = MC_coor(N,Nstep,Nmove,T[key2],x[key2],y[key2],z[key2],V[key2])
    V_2d[key1].append(V[key1])
    V_2d[key2].append(V[key2])

    re_accrate = counter*100/num_ex   # This is the acceptance rate for replica exchange
  
  return x,y,z,V,V_2d,re_accrate
