import numpy as np
from matplotlib import pyplot as plt
import pyqg

m = pyqg.QGModel(tavestart=0,  dt=8000, nx=128 // 1, ny=128 // 1, L=128*1e4)
print(m.du)
for snapshot in m.run_with_snapshots(
        tsnapstart=0, tsnapint=100*m.dt):
    plt.clf()
    plt.imshow(m.q[1])
    # plt.clim([0,  m.Qy1 * m.W])
    plt.pause(0.01)
    plt.draw()
    
# now the model is done
