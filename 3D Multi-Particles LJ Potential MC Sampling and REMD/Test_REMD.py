from REMD_simulation import replica_ex


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

Nstep = 100
Nmove = 5
T = [300,340,388,446,506,580,671]
num_ex = 90    # do num_ex times replica exchanges
x,y,z,V,V_2d,re_accrate = replica_ex(N,Nstep,Nmove,T,num_ex)    # call replica_ex fn
print('The acceptance rate of replica exchange is ', re_accrate, '%')

plt.figure(figsize=[5,5])
for i in range(len(T)):
  length = len(V_2d[i])
  Nlist = []
  for j in range(length):
    Nlist.append(100*(j+1))     # record the Nstep sequence every 100 steps
  plt.plot(Nlist,V_2d[i],label='Temp=%s'%T[i])   # plot total potential trajectory vs Nstep at T[i] temperature

plt.legend()
plt.xlabel('Nsteps')
plt.ylabel('Total LJ-potential of the box')
plt.ylim(-500,10000)
plt.title('Fig 8. Total LJ potential vs Nsteps at different temp',pad=20)
