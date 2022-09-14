### create the trajectory of MC simulation ###
from google.colab import files
from Monte_Carlo_Simulation import decide_LJ


f = open('MC_simulation_trj.xyz','w')
Nstep = 100
Nmove = 2
xlist, ylist, zlist, V = init_coor(N)
f.write('50\n')
f.write('comment\n')
for i in range(N):
  f.write('H   %s   %d   %f \n'%(xlist[i],ylist[i],zlist[i]))     # print the initial coordinates (imaging all of the points are hydrogen)

counter = 0
for i in range(Nstep):                   # print coordinates for each step
  klist = []
  while len(klist) < Nmove:              # ramdomly pick Nmove=2 keys and save in klist
    k = randrange(N)
    if k not in klist: klist.append(k)
  xlist_t = []                           # create empty trial list to save trial x coordinates for N points
  ylist_t = []
  zlist_t = []
  for j in range(N):                     # for N=50 points
    if j in klist:                       # if j is the picked Nmove points, make a move
      deltax = random.uniform(-dx, dx)
      deltay = random.uniform(-dy, dy)
      deltaz = random.uniform(-dz, dz)
      x_t = xlist[j] + deltax
      y_t = ylist[j] + deltay
      z_t = zlist[j] + deltaz
      x_t,y_t,z_t = fix_range(x_t, y_t, z_t)
    else:                                # if j is not the picked Nmove points, do not change its coordinates
      x_t = xlist[j]
      y_t = ylist[j]
      z_t = zlist[j]
    xlist_t.append(x_t)                  # add the coordinates of jth point to the trial list
    ylist_t.append(y_t)
    zlist_t.append(z_t)
  V_t = V_LJ(xlist_t, ylist_t, zlist_t)  # calculate the trial potential for trial coordinates
  V, xlist, ylist, zlist, counter = decide_LJ(V_t, V, xlist_t, ylist_t, zlist_t, xlist, ylist, zlist, T, counter)  # decide if taking the move (N=50,Nstep=100,Nmove=2,T=300K)
  f.write('50\n')
  f.write('comment\n')
  for i in range(N):
    f.write('H   %s   %d   %f \n'%(xlist[i],ylist[i],zlist[i]))  # write the coordinated for ith MC step.

f.close()
files.download('MC_simulation_trj.xyz')   # download the trajectory file
