# This program generates NX initial conditions, sampling from a classical (MM) molecular dynamics trajectory of a solute in an explicit solvent, performed with PBC. It is assumed that the solute molecule is the first residue. The MD trajectory is in binary .dcd format; coordinates are in .dcd file (angstrom), velocities in .veldcd file (angstrom ps^-1).
# F. Perrella 21/03/2026


import yaml
import sys
import subprocess
import numpy as np
import scipy
import MDAnalysis as mda
from MDAnalysis import transformations


Z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'I':53}
geomfile_format = '{:2}   {:4.1f}   {:13.8f}   {:13.8f}   {:13.8f}   {:13.8f}\n'
velocfile_format = ' {:14.9f}   {:14.9f}   {:14.9f}\n'
gaufile_format = ' {:14}      {:13.8f}   {:13.8f}   {:13.8f}   {:1}\n'
Eh = scipy.constants.physical_constants['Hartree energy'][0]
a0 = scipy.constants.physical_constants['Bohr radius'][0]
hbar = scipy.constants.hbar
conv_f_coor = (1.e-10)*(a0**(-1))
conv_f_vel = 100.*(a0**(-1))*hbar*(Eh**(-1))
conv_vel_namd = 20.45482706  # from namd vel units to A ps^-1

input_file = sys.argv[1]
with open(input_file,'r') as f:
  inp = yaml.load(f,Loader=yaml.FullLoader)

solute_params = []
with open(inp['solute_frcmod'],'r') as f:
  line = ''
  while line[0:4] != 'NONB':
    line = f.readline().rstrip()
  while line != '':
    line = f.readline().rstrip()
    if line != '':
      line_spl = line.split()
      solute_params.append('VDW {:3}   {:6.4f}   {:6.4f}'.format(line_spl[0]+'X',float(line_spl[1]),float(line_spl[2])))

solvent_params = []
with open(inp['solvent_frcmod']) as f:
  while line != 'BOND':
    line = f.readline().rstrip()
  while line != '':
    line = f.readline().rstrip()
    if line != '':
      line = line.replace('-',' ')
      line_spl = line.split()
      solvent_params.append('HrmStr1 {:3} {:3}   {:5.1f}   {:5.3f}'.format(line_spl[0],line_spl[1],float(line_spl[2]),float(line_spl[3])))
  while line != 'ANGL':
    line = f.readline().rstrip()
  while line != '':
    line = f.readline().rstrip()
    if line != '':
      line = line.replace('-',' ')
      line_spl = line.split()
      solvent_params.append('HrmBnd1 {:3} {:3} {:3}   {:5.1f}   {:6.2f}'.format(line_spl[0],line_spl[1],line_spl[2],float(line_spl[3]),float(line_spl[4])))
  while line != 'DIHEDR':
    line = f.readline().rstrip()
  while line != '':
    line = f.readline().rstrip()
    if line != '':
      line = line.replace('-',' ')
      line_spl = line.split()
      V2 = np.zeros(4)
      phase = np.zeros(4)
      p = int(line_spl[7])
      V2[p-1] = float(line_spl[5])
      phase[p-1] = float(line_spl[6])
      npaths = int(line_spl[4])
      solvent_params.append('AmbTrs {:3} {:3} {:3} {:3}   {:4d} {:4d} {:4d} {:4d}   {:6.3f} {:6.3f} {:6.3f} {:6.3f}   {:3.1f}'.format(line_spl[0],line_spl[1],line_spl[2],line_spl[3],int(phase[0]),int(phase[1]),int(phase[2]),int(phase[3]),V2[0],V2[1],V2[2],V2[3],float(npaths)))
      solvent_params[-1] = solvent_params[-1].replace('X','*')
  while line[0:4] != 'NONB':
    line = f.readline().rstrip()
  while line != '':
    line = f.readline().rstrip()
    if line != '':
      line_spl = line.split()
      solvent_params.append('VDW {:3}   {:6.4f}   {:6.4f}'.format(line_spl[0],float(line_spl[1]),float(line_spl[2])))

u = mda.Universe(inp['top_file'],inp['coor_file'],format='dcd')
uvel = mda.Universe(inp['top_file'],inp['vel_file'],format='dcd')

solute_ag = u.select_atoms('resnum 1')
solvent_ag = u.select_atoms('not resnum 1')
transf = [transformations.unwrap(u.atoms),transformations.center_in_box(solute_ag,center='geometry'),transformations.wrap(solvent_ag,compound='residues')]
u.trajectory.add_transformations(*transf)

NFrames = len(u.trajectory)
Nres = u.atoms.n_residues
Nat_solute = solute_ag.n_atoms
Nat_solvent = u.select_atoms('resnum 10').n_atoms
Ninit = int(inp['Ninit'])
radius = float(inp['radius'])
inner_radius = float(inp['inner_radius'])
nxinput = inp['nx_input'].split('\n')

rng = np.random.default_rng()
i_sel = rng.integers(low=0,high=NFrames,size=Ninit)

Nsolv_frames = []
for i in i_sel:
  u.trajectory[i]
  solute_cog = solute_ag.center_of_geometry()
  NN = 0
  for r in range(2,Nres+1):
    solv_mol = u.select_atoms('resnum '+str(r))
    solv_cog = solv_mol.center_of_geometry()
    if np.linalg.norm(solv_cog-solute_cog) < radius:
      NN += 1
  Nsolv_frames.append(NN)
Nsolv_in = min(Nsolv_frames)

subprocess.run(['mkdir','TRAJECTORIES'])
j = 0
for i in i_sel:
  j += 1
  folder = 'TRAJECTORIES/'+'TRAJ'+str(j)
  subprocess.run(['mkdir','-p',folder+'/JOB_AD'])
  u.trajectory[i]
  uvel.trajectory[i]
  Nadded = 0  # number of atoms retained
  Nsolvadd = 0  # number of solvent molecules retained

  with open(folder+'/user_config.nml','w') as nxfile, open(folder+'/geom','w') as geomfile, open(folder+'/veloc','w') as velocfile, open(folder+'/freeze.inp','w') as freezefile, open(folder+'/JOB_AD'+'/gaussian.com','w') as gaufile, open(folder+'/JOB_AD'+'/basis','w') as basisfile:
    gaufile.write(inp['gau_input'])
    basisfile.write(inp['basis'])
    solute_cog = solute_ag.center_of_geometry()
    solute_coor = solute_ag.positions - solute_cog
    solute_vel = uvel.select_atoms('resnum 1').positions
    for a in range(Nat_solute):
      geomfile.write(geomfile_format.format(solute_ag.elements[a],Z[solute_ag.elements[a]],solute_coor[a,0]*conv_f_coor,solute_coor[a,1]*conv_f_coor,solute_coor[a,2]*conv_f_coor,solute_ag.masses[a]))
      velocfile.write(velocfile_format.format(solute_vel[a,0]*conv_vel_namd*conv_f_vel,solute_vel[a,1]*conv_vel_namd*conv_f_vel,solute_vel[a,2]*conv_vel_namd*conv_f_vel)) 
      gaufile.write(gaufile_format.format(solute_ag.elements[a]+'-'+solute_ag.types[a]+'X'+'-'+'{:.4f}'.format(solute_ag.charges[a]),solute_coor[a,0],solute_coor[a,1],solute_coor[a,2],'H'))
    Nadded += Nat_solute
    for r in range(2,Nres+1):
      solv_mol = u.select_atoms('resnum '+str(r))
      solv_cog = solv_mol.center_of_geometry()
      if np.linalg.norm(solv_cog-solute_cog) < radius:
        solv_coor = solv_mol.positions - solute_cog
        solv_vel = uvel.select_atoms('resnum '+str(r)).positions
        for a in range(Nat_solvent):
          geomfile.write(geomfile_format.format(solv_mol.elements[a],Z[solv_mol.elements[a]],solv_coor[a,0]*conv_f_coor,solv_coor[a,1]*conv_f_coor,solv_coor[a,2]*conv_f_coor,solv_mol.masses[a]))
          velocfile.write(velocfile_format.format(solv_vel[a,0]*conv_vel_namd*conv_f_vel,solv_vel[a,1]*conv_vel_namd*conv_f_vel,solv_vel[a,2]*conv_vel_namd*conv_f_vel)) 
          gaufile.write(gaufile_format.format(solv_mol.elements[a]+'-'+solv_mol.types[a]+'-'+'{:.4f}'.format(solv_mol.charges[a]),solv_coor[a,0],solv_coor[a,1],solv_coor[a,2],'L'))
        if np.linalg.norm(solv_cog-solute_cog) > inner_radius:
          freezefile.write(str(Nadded+1)+'-'+str(Nadded+Nat_solvent)+'\n')
        Nadded += Nat_solvent
        Nsolvadd += 1
        if Nsolvadd == Nsolv_in:
          break
    gaufile.write('\n')
    for param in solute_params:
      gaufile.write(param+'\n')
    for param in solvent_params:
      gaufile.write(param+'\n')
    gaufile.write('\n\n')
    nxfile.write(nxinput[0]+'\n')
    nxfile.write('  nat = '+str(Nat_solute+Nsolv_in*Nat_solvent)+'\n')
    for line in nxinput[1:]:
      nxfile.write(line+'\n')
  subprocess.run(['sed','-i','s/chX/ciX/',folder+'/JOB_AD'+'/gaussian.com'])  # Gaussian doesn't like atom types with ch


