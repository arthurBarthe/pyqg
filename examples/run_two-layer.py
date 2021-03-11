import numpy as np
from matplotlib import pyplot as plt
import pyqg

m = pyqg.QGModel(tavestart=0,  dt=8000, nx=256 // 1, L = 1e6,
                 filterfac=23.6)

for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=100*m.dt):
    plt.clf()
    plt.imshow(m.q[0] + m.Qy1 * m.y)
    plt.clim([0, m.Qy1 * m.W])
    plt.colorbar()
    plt.pause(0.01)
    plt.draw()
# now the model is done
