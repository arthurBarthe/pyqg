import numpy as np
from matplotlib import pyplot as plt
import pyqg
from pathlib import Path

from utils import energy_budget

path_output_dir = Path('/media/arthur/DATA/Data sets/outputs/pyqg/')
size = 256 // 4
year = 365 * 24 * 3600
m = pyqg.QGModel(tavestart=10 * year, dt=8000 / 2, nx=size,
                 L = 1e6)

snapshots = dict(q=[], u=[], v=[])
for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=2000*m.dt):
    for var in ('q', 'u', 'v'):
        arr = np.asarray(getattr(m, var).copy())
        snapshots[var].append(arr)
energy_budget(m)
plt.show()
# now the model is done

for var in ('q', 'u', 'v'):
    video = np.stack(snapshots[var], axis=0)
    assert not np.all(video[0, ...] == video[10, ...])
    np.save(path_output_dir / f'video_{var}_{size}', video)

energy_budget(m, path_output_dir, size)
