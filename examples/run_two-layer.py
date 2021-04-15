import numpy as np
from matplotlib import pyplot as plt
import pyqg
from pathlib import Path
import logging
from utils import energy_budget, coarsen

path_output_dir = Path('/media/arthur/DATA/Data sets/outputs/pyqg/')
size = 256
year = 365 * 24 * 3600
m = pyqg.QGModel(tavestart=10 * year, dt=8000 / 6, nx=size,
                 L = 2e6, U1=0.05)

print(m.dx)
print(m.rd)
plt.imshow(m.u[0], vmin=-1, vmax=1)
plt.colorbar()
snapshots = dict(q=[], u=[], v=[])
for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=2000*m.dt):
    for var in ('q', 'u', 'v'):
        arr = np.asarray(getattr(m, var).copy())
        snapshots[var].append(arr)
    u_plot = m.u[0]
    plt.imshow(u_plot, vmin=-1, vmax=1)
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
