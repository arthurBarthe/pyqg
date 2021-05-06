import numpy as np
from matplotlib import pyplot as plt
import pyqg
from pathlib import Path
import logging
from utils import energy_budget, coarsen
import os
import pickle

experiment_name = 'Trash'

path_output_dir = Path('/media/arthur/DATA/Data sets/outputs/pyqg/')
path_output_dir = path_output_dir / experiment_name
if not path_output_dir.exists():
    path_output_dir.mkdir()

size = 64
year = 365 * 24 * 3600
day = 3600 * 24
m = pyqg.QGModel(tavestart=1 * year, tmax=2 * year, dt=8000 / 2, nx=size,
                 L=1e6, U1=0.025)

print(m.dx)
print(m.rd)
plt.imshow(m.u[0], vmin=-0.5, vmax=0.5)
plt.colorbar()
snapshots = dict(q=[], u=[], v=[])
for snapshot in m.run_with_snapshots(
        tsnapstart=15 * year, tsnapint=5 * day):
    for var in ('q', 'u', 'v'):
        arr = np.asarray(getattr(m, var).copy())
        snapshots[var].append(arr)
    u_plot = m.u[0]
    plt.imshow(u_plot, vmin=-0.5, vmax=0.5)
    plt.pause(0.01)
    print(np.max(u_plot))
    try:
        print("EKE: ", m.get_diagnostic('EKE'))
    except:
        logging.debug('EKE not available yet')
energy_budget(m)
plt.show()
# now the model is done

for var in ('q', 'u', 'v'):
    video = np.stack(snapshots[var], axis=0)
    assert not np.all(video[0, ...] == video[10, ...])
    np.save(path_output_dir / f'video_{var}_{size}', video)

energy_budget(m, path_output_dir, size)