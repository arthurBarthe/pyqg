import numpy as np
from matplotlib import pyplot as plt
import pyqg

from utils import energy_budget

m = pyqg.QGModel(tavestart=0,  dt=8000 / 2, nx=256 // 4, L = 1e6,
                 filterfac=23.6)

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=1000*m.dt):
    plt.clf()
    plt.imshow(m.q[0] + m.Qy1 * m.y)
    plt.clim([0, m.Qy1 * m.W])
    plt.colorbar()
    plt.pause(0.01)
    plt.draw()
    print("EKE: ", m.get_diagnostic('EKE'))
# now the model is done

energy_budget(m)
plt.show()