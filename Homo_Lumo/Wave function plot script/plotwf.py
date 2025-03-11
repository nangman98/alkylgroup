import os
import sisl
import numpy as np
import multiprocessing as mp
from functools import partial

# Plot SIESTA wavefunctions in real space
# 1. Run SIESTA to get .WFSX file
# (Example input)
# WriteWaveFunctions T
# WaveFuncKPointsScale ReciprocalLatticeVectors
# %block WaveFuncKPoints
# 0.0 0.0 0.0 from 200 to 300
# %endblock WaveFuncKPoints
#
# 2. Run this script
# python plot_wavefunctions.py
#
# 3. Open .xsf files using VESTA
#

# Input fdf file
fdffile = './wave_input.fdf'
# WFSX file
wfsxfile = './siesta.selected.WFSX'
# Grid shape (refer sisl.Grid doc)
grid_shape = 0.1
# Output directory
outdir = 'wfplot'
# Prefix for output files
prefix = 'siesta'
# Target eigenstates (None for all)
# target = list(range(0, 416))
target = [0,1] #None

# Number of parallel processes
num_pools = 16

os.makedirs(outdir, exist_ok=True)

fdf = sisl.io.siesta.fdfSileSiesta(fdffile)
geom = fdf.read_geometry()
wfsx = sisl.io.siesta.wfsxSileSiesta(wfsxfile, parent=geom)
es = wfsx.read_eigenstate()
neig = len(es)

print(f'Found {neig} eigenstates')
print(f'Writing to {outdir}/{prefix}_*.xsf')
if target is not None:
    print(f'Target eigenstates: {target}')
else:
    print('Target eigenstates: all')

if target is None:
    ieig_selected = np.arange(neig)
else:
    ieig_selected = target.copy()


def process_wavefunction(i, es, grid_shape, geom, outdir, prefix):
    print(f'Processing {i}')
    grid = sisl.Grid(grid_shape, lattice=geom.cell)
    es[i].wavefunction(grid)
    grid.write(f'{outdir}/{prefix}_{i}.xsf')


with mp.Pool(processes=num_pools) as pool:
    pool.map(partial(process_wavefunction, es=es, grid_shape=grid_shape,
             geom=geom, outdir=outdir, prefix=prefix), ieig_selected)

print('Done!')