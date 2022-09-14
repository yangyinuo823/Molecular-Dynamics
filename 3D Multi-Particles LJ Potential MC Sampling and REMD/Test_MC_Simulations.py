import random
from random import randrange
import numpy as np
import time
import sympy as sym
import math
import scipy as sci
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as mplpyl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Monte_Carlo_Simulation import MC_coor

### Default values ###
kb = 1.98*(10**(-3))          # Boltzmann constant in kcal/mol.K
eps = 3                       # units in kcal/mol
sigma = 2                     # units in angstrom (for LJ potential)
T = 300                       # units in kelvin
N = 50                        # number of points within the box
Nstep = 100                   # number of steps
minx   =  -10                 # Minimum of x
maxx   =  10                  # Maximum of x
rangex = maxx - minx          # Boxsize on x dimension
miny = -10
maxy = 10
rangey = maxy - miny
minz = 0
maxz = 0
rangez = maxz - minz         # z-dimension is listed here, but I make the range of z as 0, so that it is still a two-dimention system
dx = 2
dy = 2
dz = 0 
cutoff = 9                   # cutoff < boxsize/2


### Test1: Intuitively test the MC simulation (N=50,Nstep=1000,Nmove=1,T=300K,dx=dy=2)
Nstep = 1000
Nmove = 1            # do MC move for only one point in each step
fig, axs = plt.subplots(1, 2, figsize=(15,6.5))
axs = axs.ravel()
x_ini, y_ini, z_ini, V_ini = init_coor(N)         # get the initial coordinates and potential for a box with default size (20*20*0, N=50 points)
axs[0].scatter(x_ini,y_ini)
axs[0].set_title('Fig 0-a. Before MC simulation')
x_MC, y_MC, z_MC, V_MC, acc_rate = MC_coor(N,Nstep,Nmove,T,x_ini,y_ini,z_ini,V_ini)    # Do Nstep=1000 MC steps, get the final coordiantes/potential/acceptance rate (default stepsize dx=dy=2, T=300K)
axs[1].scatter(x_MC,y_MC)
axs[1].set_title('Fig 0-b. After MC simulation with Nstep=1000')
print('The initial LJ potential is ', V_ini)
print('The final LJ potential is ', V_MC)


### Test2: Test the MC moves (N=50,Nstep=200,T=300K)
# (a) move one point at a time
Nmove = 1
Nstep = 200
stepsize = [0.2,0.5,1,3,5,7,10]     
x_ini, y_ini, z_ini, V_ini = init_coor(N)   # get initial coordinates/potential for N=50 points (boxsize:20*20*0)
plt.figure(figsize=[5,5])
for i in range(5):    # repeat for 5 runs
  acc_list = []       
  for dx,dy in zip(stepsize,stepsize):    # test for different stepsize(dx/dy), change dx/dy simultaneously
    x_MC,y_MC,z_MC,V_MC,acc_rate = MC_coor(N,Nstep,Nmove,T,x_ini,y_ini,z_ini,V_ini)    # Do MC for Nstep=200 times with Nmove=1, T=300K
    acc_list.append(acc_rate)        # save acceptance rate changes with stepsize
  plt.plot(stepsize, acc_list, label='run %s'%(i+1))     # plot acceptance rate vs stepsize
plt.legend()
plt.xlabel('stepsize for dx/dy (angstrom)')
plt.ylabel('acceptance rate (%)')
plt.title('Fig 2. Acceptance rate vs stepsize (Nmove=1)')

# (b) move multiple points at a time
Nstep = 200
Nmove_list = [1,5,10,15,20,30,50]
x_ini, y_ini, z_ini, V_ini = init_coor(N)       # get initial coordinates/potential for N=50 points (boxsize:20*20*0)
plt.figure(figsize=[5,5])
for i in range(5):     # repeat for 5 runs
  acc_list = []
  for Nmove in Nmove_list:      # test for different Nmove(num of points that moves during one MC step)
    x_MC,y_MC,z_MC,V_MC,acc_rate = MC_coor(N,Nstep,Nmove,T,x_ini,y_ini,z_ini,V_ini)   # do MC for diff Nmoves (N=50 points, Nstep=200, T=300K)
    acc_list.append(acc_rate)
  plt.plot(Nmove_list, acc_list, label='run %s'%(i+1))
plt.legend()
plt.xlabel('Number of points moves at one MC step')
plt.ylabel('Acceptance rate(%)')
plt.title('Fig 3. Acceptance rate vs Nmoves')


### Test3: Test the effect of temperature and epsilon
Nstep = 1000
Nmove = 5
T_list = [10,1000,10000,50000,100000]
eps_list = [1,50,100,200,300]
x_ini, y_ini, z_ini, V_ini = init_coor(N)     # get initial coordinates/potential for N=50 points (boxsize:20*20*0)
plt.figure(figsize=[5,5])
for eps in eps_list:         # test for different epsilon
  acc_list = []
  for T in T_list:           # test for different Temp
    x_MC,y_MC,z_MC,V_MC,acc_rate = MC_coor(N,Nstep,Nmove,T,x_ini,y_ini,z_ini,V_ini)      # do MC (N=50 points, Nmove=5, Nstep=1000)
    acc_list.append(acc_rate)
  plt.plot(T_list, acc_list, label='eps=%s'%(eps))
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('Acceptance rate(%)')
plt.title('Fig 4. Acceptance rate vs Temp with different epsilon')



### Test4: Test the convergence of the MC simulation (ergodicity)
# Convergence of MC simulation (no replica exchange) ###
# (a) average total potential vs Nstep ###
def MC_avg_V(N,Nstep,Nmove,T,xlist,ylist,zlist,V):
  '''
  This fn is used to track the average total potential and plot it vs Nstep
  This fn is almost the same as the MC_coor function defined above. 
  (The only difference is it records the average total potential trajectory)
  '''
  counter = 0                             # used for acceptance rate calculation
  Nc = 1                                  # used for counting the steps
  Vlist = [V]                             # save total potential trajectory
  avg_Vlist = [V]                         # save average total potential with Nstep (ergodicity)
  Nlist = [Nc]                            # save 1,2,3,4,....,Nstep
  for i in range(Nstep):
    klist = []
    while len(klist) < Nmove:             # ramdomly pick Nmove keys and save in klist
      k = randrange(N)
      if k not in klist: klist.append(k)
    xlist_t = []
    ylist_t = []
    zlist_t = []
    for j in range(N):            
      if j in klist:
        deltax = random.uniform(-dx, dx)
        deltay = random.uniform(-dy, dy)
        deltaz = random.uniform(-dz, dz)
        x_t = xlist[j] + deltax
        y_t = ylist[j] + deltay
        z_t = zlist[j] + deltaz
        x_t,y_t,z_t = fix_range(x_t, y_t, z_t)
      else:
        x_t = xlist[j]
        y_t = ylist[j]
        z_t = zlist[j]
      xlist_t.append(x_t)
      ylist_t.append(y_t)
      zlist_t.append(z_t)
    V_t = V_LJ(xlist_t, ylist_t, zlist_t)
    V, xlist, ylist, zlist, counter = decide_LJ(V_t, V, xlist_t, ylist_t, zlist_t, xlist, ylist, zlist, T, counter)
    Nc += 1
    Nlist.append(Nc)
    Vlist.append(V)
    avg_Vlist.append(sum(Vlist)/len(Vlist))
  return avg_Vlist, Nlist

Nstep = 5000
Nmove = 5
x_ini, y_ini, z_ini, V_ini = init_coor(N)
plt.figure(figsize=[5,5])
for i in range(5):
  avg_Vlist, Nlist = MC_avg_V(N,Nstep,Nmove,T,x_ini,y_ini,z_ini,V_ini)
  plt.plot(Nlist,avg_Vlist,label='run %s'%(i+1))
plt.legend()
plt.xlabel('Nsteps')
plt.ylabel('Avgerage LJ potential of the box')
plt.title('Fig 5. Average total potential vs Nsteps for multiple runs',pad=20)


# (b) average potential for each particle  ###
def V_LJ_one(xlist,ylist,zlist,key):
  '''
  This fn is used to calculate the potential between the keyth point and other points 
  '''
  V_one = 0
  for i in range(N):
    if i == key:
      V_one += 0
    else:
      r = distance(xlist[i], ylist[i], zlist[i], xlist[key], ylist[key], zlist[key])   
      if r < cutoff:
        v = 4*eps*((sigma/r)**12 - (sigma/r)**6)
        V_one += v
  return V_one

def MC_avg_Vone(N,Nstep,Nmove,T,xlist,ylist,zlist,V):
  '''
  This fn is used to track the average potential for each particle
  This fn is almost the same as the MC_coor function defined above. 
  (The only difference is it records the total potential for each point)
  '''
  counter = 0                             # used for acceptance rate calculation
  Nc = 0                                  # used to count the steps
  Nlist = []                              # save 1,2,3,4,....,Nstep
  Vone_traj = [[] for i in range(N)]      # 2d-array, Vone_traj[i][j] is the ith point's potential at jth MC step
  avg_Vone_traj = [[] for i in range(N)]  # 2d-array, avg_Vone_traj[i][j] is the ith point's average potential calculated at jth MC step
  for i in range(Nstep):                  
    klist = []
    while len(klist) < Nmove:             # ramdomly pick Nmove keys and save in klist
      k = randrange(N)
      if k not in klist: klist.append(k)
    xlist_t = []
    ylist_t = []
    zlist_t = []
    for j in range(N):            
      if j in klist:
        deltax = random.uniform(-dx, dx)
        deltay = random.uniform(-dy, dy)
        deltaz = random.uniform(-dz, dz)
        x_t = xlist[j] + deltax
        y_t = ylist[j] + deltay
        z_t = zlist[j] + deltaz
        x_t,y_t,z_t = fix_range(x_t, y_t, z_t)
      else:
        x_t = xlist[j]
        y_t = ylist[j]
        z_t = zlist[j]
      xlist_t.append(x_t)
      ylist_t.append(y_t)
      zlist_t.append(z_t)
    V_t = V_LJ(xlist_t, ylist_t, zlist_t)
    V, xlist, ylist, zlist, counter = decide_LJ(V_t, V, xlist_t, ylist_t, zlist_t, xlist, ylist, zlist, T, counter)
    Nc += 1
    Nlist.append(Nc)
    for key in range(N):                  # record the potential and average potential for each point at each step(Vone_traj & avg_Vone_traj)
      Vone = V_LJ_one(xlist,ylist,zlist,key)
      Vone_traj[key].append(Vone)
      avg_Vone_traj[key].append(sum(Vone_traj[key])/len(Vone_traj[key]))

  return Nlist, avg_Vone_traj             # return Nlist(1,2...Nstep) and average potential trajectory for each point

Nstep = 10000                             # Nstep=10000 is the longest num of steps I have tried. It takes a lot of time
Nmove = 5                                 # pick Nmove=5
x_ini, y_ini, z_ini, V_ini = init_coor(N) # generate initial coordinates and potential (default N=50, boxsize:20*20*0, stepsize:dx=dy=2)
Nlist, avg_Vone_traj = MC_avg_Vone(N,Nstep,Nmove,T,x_ini,y_ini,z_ini,V_ini)   #Do MC and return the average potential trajectory for each points
plt.figure(figsize=[5,5])
for key in range(N):
  plt.plot(Nlist,avg_Vone_traj[key])
plt.xlabel('Nsteps')
plt.ylabel('Average LJ-potential')
plt.ylim(0,1e5)
plt.title('Average LJ-potential vs Nsteps for each particle (ergodicity)',pad=20)


# (c) To make it easier, plot total potential vs Nstep  ###
def MC_V(N,Nstep,Nmove,T,xlist,ylist,zlist,V):
  '''
  This fn is used to track the average total potential and plot it vs Nstep
  This fn is almost the same as the MC_coor fn defined above. 
  (The only difference is it records the total potential trajectory (not average total potential))
  '''
  counter = 0                               # used for acceptance rate calculation
  Nc = 1                                    # used for counting the steps
  Vlist = [V]                               # save total potential trajectory
  Nlist = [Nc]                              # save 1,2,3,4,....,Nstep
  for i in range(Nstep):
    klist = []                              # ramdomly pick Nmove keys and save in klist
    while len(klist) < Nmove:
      k = randrange(N)
      if k not in klist: klist.append(k)
    xlist_t = []
    ylist_t = []
    zlist_t = []
    for j in range(N):            
      if j in klist:
        deltax = random.uniform(-dx, dx)
        deltay = random.uniform(-dy, dy)
        deltaz = random.uniform(-dz, dz)
        x_t = xlist[j] + deltax
        y_t = ylist[j] + deltay
        z_t = zlist[j] + deltaz
        x_t,y_t,z_t = fix_range(x_t, y_t, z_t)
      else:
        x_t = xlist[j]
        y_t = ylist[j]
        z_t = zlist[j]
      xlist_t.append(x_t)
      ylist_t.append(y_t)
      zlist_t.append(z_t)
    V_t = V_LJ(xlist_t, ylist_t, zlist_t)
    V, xlist, ylist, zlist, counter = decide_LJ(V_t, V, xlist_t, ylist_t, zlist_t, xlist, ylist, zlist, T, counter)
    Nc += 1
    Nlist.append(Nc)
    Vlist.append(V)                         # record the total potential trajectory
  return Vlist, Nlist

Nstep = 10000
Nmove = 5
x_ini, y_ini, z_ini, V_ini = init_coor(N)   # generate initial coordinates and potential (default N=50, boxsize:20*20*0, stepsize:dx=dy=2)
plt.figure(figsize=[5,5])
for i in range(5):      # repeat for 5 runs
  Vlist, Nlist = MC_V(N,Nstep,Nmove,T,x_ini,y_ini,z_ini,V_ini)   # Do MC and return the total potential trajectory (N=50,T=300K)
  plt.plot(Nlist,Vlist,label='run %s'%(i+1))
plt.legend()
plt.xlabel('Nsteps')
plt.ylabel('LJ potential of the box')
plt.ylim(-500,10000)
plt.title('Fig 7. Total LJ potential vs Nsteps for multiple runs',pad=20)
