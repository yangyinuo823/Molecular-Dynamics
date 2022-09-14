##Set up Monte Carlo Simulation

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

%matplotlib inline 


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

#### define functions
def dis_1(x_t, x, rangex): 
  '''
  This function is used to get the periodic distance on one dimension
  x, x_t: the one dimension coordinate of two points x & x_t
  rangex: the boxsize on this dimension
  This function works by returning the minimum of one dimension two point distances (both in-box distance & cross-box distance)
  '''
  xa = np.abs(x_t - x)
  xb = rangex - xa
  return min(xa, xb)


def distance(x_t, y_t, z_t, x, y, z): 
  '''
  This function is used to get the periodic distance between two points on three dimensions
  '''
  xd = dis_1(x_t, x, rangex)
  yd = dis_1(y_t, y, rangey)
  zd = dis_1(z_t, z, rangez)
  r = np.sqrt(xd**2 + yd**2 + zd**2)
  return r


def V_LJ(xlist, ylist, zlist): 
  '''
  This function is used to calculate the total LJ potential of N pts in a periodical box
  The LJ potential is applied within a cutoff
  '''  
  V = 0                                
  for i in range(N):        # N points, N=50
    for j in range(i):
      r = distance(xlist[i], ylist[i], zlist[i], xlist[j], ylist[j], zlist[j])    # calculate distance between ith point & jth point
      if r < cutoff:        # Only consider distance smaller than cutoff
        v = 4*eps*((sigma/r)**12 - (sigma/r)**6)
        V += v              # Cumulate all the inter-points potentials within cutoff
  return V


def get_rand(min, max, sd=time.time()):  # generate a random float value between min and max
  random.seed(sd)
  randnum = random.uniform(min, max)
  return randnum



def get_boltz(dV, T):         
  '''
  This function determines the probability of accepting or rejecting a move according to a Boltzmann distribution
  dV = Difference in potential between the previous and trial state
  '''
  beta = 1/(kb*T)            
  boltz = np.exp(-beta*dV)
  return boltz


def fix_range1(x, maxx, minx):     
  '''
  This function is used to check if a move is cross a box on x dimension.
  If true, send this point back within the x range
  '''
  if x > maxx:
    x -= (maxx - minx)
  elif x <= minx:
    x += (maxx - minx)
  return x

def fix_range(x, y, z):
  '''
  This function is used to check if a move is cross a box on three dimensions.
  If true, send this point back to the inside of the box
  '''
  x = fix_range1(x, maxx, minx)
  y = fix_range1(y, maxy, miny)
  z = fix_range1(z, maxz, minz)
  return x, y, z


def decide_LJ(V_t, V, x_t, y_t, z_t, x, y, z, T, counter):
  '''
  This is the decide function for LJ potential, it is used to accept/reject a move
  x_t,y_t,z_t,x,y,z are all lists of all N points inside the box
  '''
  prob = get_rand(0,1,time.time())
  dV = (V_t - V) 
  boltz = get_boltz(dV,T)     # calculate boltzmann distribution probability
  if prob < boltz:            # accept a move   
    x = x_t
    y = y_t
    z = z_t
    V = V_t
    counter += 1              # 'counter' is used for calcuating acceptance rate
  else:                       # reject a move
    x = x
    y = y
    z = z
    V = V
  return V, x, y, z, counter # return decided potential/coordinates and number of accepted moves


def init_coor(N): 
  '''
  This function is used to initiate the coordinates of N data points
  All the coordinates are randomly picked within the boxsize
  It returns all the coordinates and the potential V
  '''
  xlist = []                       # create empty list to save x coordinates
  ylist = []
  zlist = []
  for i in range(N):
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    z = random.uniform(minz, maxz)
    xlist.append(x)
    ylist.append(y)
    zlist.append(z)
  V = V_LJ(xlist, ylist, zlist)    # call the potential function V_LJ for the total potential within a box
  return xlist,ylist,zlist,V



def MC_coor(N,Nstep,Nmove,T,xlist,ylist,zlist,V):   # N: num of points in box, Nstep: num of MC steps, Nmove: num of points doing MC move in one step
  '''
  This function is used to do Metropolis Monte Carlo to a random subset of N points (Nmove points)
  For each step, the random Nmove points will take a random move on x/y/z dimensions, decide_LJ function will be used here to accept/reject a move
  '''
  counter = 0

  for i in range(Nstep):                   # Do MC for Nstep iterations

    klist = []                             # generate random keys between 0~49, save these keys in klist
    while len(klist) < Nmove:
      k = randrange(N)
      if k not in klist: klist.append(k)

    xlist_t = []                           # create empty trial list to save trial x coordinates for N points
    ylist_t = []
    zlist_t = []
    for j in range(N):                     # for N points, get the trial coordinates for each point
      if j in klist:                       # if j belongs to klist (randomly picked Nmove points), take a MC move for each of them
        deltax = random.uniform(-dx, dx)
        deltay = random.uniform(-dy, dy)
        deltaz = random.uniform(-dz, dz)
        x_t = xlist[j] + deltax            # x_t is the trial x coordinate for jth point 
        y_t = ylist[j] + deltay
        z_t = zlist[j] + deltaz
        x_t,y_t,z_t = fix_range(x_t, y_t, z_t)   # check if the point is outside of box, if yes return it back
      else:
        x_t = xlist[j]                     # if j do not belong to klist, keep their original coordinates
        y_t = ylist[j]
        z_t = zlist[j]
      xlist_t.append(x_t)                  # Put all the trial coordinates into the trial list for N points
      ylist_t.append(y_t)
      zlist_t.append(z_t)

    V_t = V_LJ(xlist_t, ylist_t, zlist_t)  # calculate the trial potential for the trial coordinates
    V, xlist, ylist, zlist, counter = decide_LJ(V_t, V, xlist_t, ylist_t, zlist_t, xlist, ylist, zlist, T, counter)    # decide if accept this MC move
    acc_rate = counter / Nstep * 100       # acceptance rate
  return xlist,ylist,zlist,V,acc_rate
