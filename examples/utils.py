import matplotlib.pyplot as plt
import numpy as np
from pyqg.model import Model

def energy_budget(m):
    # some spectral plots
    KE1spec = m.get_diagnostic('KEspec')[0].sum(
        axis=0)  # note that this is misleading for anisotrphic flows...
    KE2spec = m.get_diagnostic('KEspec')[1].sum(
        axis=0)  # we should sum azimuthaly, and plot as a functions of kappa

    # factor ebud
    ebud_factor = 1.e4
    ebud_factor_s = str('%1.1e') % ebud_factor

    # inertial range
    ir = np.r_[10:20]

    fig = plt.figure(figsize=(16., 7.))
    ax1 = fig.add_subplot(121)
    ax1.loglog(m.kk, KE1spec, '.-')
    ax1.loglog(m.kk, KE2spec, '.-')
    ax1.loglog(m.kk[10:20], 2 * (m.kk[ir] ** -3) *
               KE1spec[ir].mean() / (m.kk[ir] ** -3).mean(),
               '0.5')
    ax1.set_ylim([1e-9, 1e-3])
    ax1.set_xlim([m.kk.min(), m.kk.max()])
    ax1.grid()
    ax1.legend(['upper layer', 'lower layer', r'$k^{-3}$'],
               loc='lower left')
    ax1.set_xlabel(r'k (m$^{-1})$')
    ax1.set_title('Kinetic Energy Spectrum')

    # the spectral energy budget
    ebud = [-m.get_diagnostic('APEgenspec').sum(axis=0),
            -m.get_diagnostic('APEflux').sum(axis=0),
            -m.get_diagnostic('KEflux').sum(axis=0),
            -m.rek * m.del2 * m.get_diagnostic('KEspec')[1].sum(
                axis=0) * m.M ** 2]
    ebud.append(-np.vstack(ebud).sum(axis=0))
    ebud_labels = ['APE gen', 'APE flux', 'KE flux', 'Diss.', 'Resid.']

    ax2 = fig.add_subplot(122)
    [ax2.semilogx(m.kk, term) for term in ebud]
    ax2.legend(ebud_labels, loc='upper right')

    ax2.grid()
    ax2.set_xlim([m.kk.min(), m.kk.max()])
    ax1.set_xlabel(r'k (m$^{-1})$')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.set_title(r'      Spectral Energy Budget')

    ax2.set_xlabel(r'k (m$^{-1})$')