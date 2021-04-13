import matplotlib.pyplot as plt
import numpy as np
import numpy
from pyqg.model import Model
from matplotlib import animation
from scipy.fft import fftshift

# Function below from
# https://github.com/fatiando/fatiando/blob/master/fatiando/gravmag/transform.py
# Fatiando project
def radial_average_spectrum(kx, ky, pds, max_radius=None, ring_width=None):
    r"""
    Calculates the average of the Power Density Spectra points that falls
    inside concentric rings built around the origin of the wavenumber
    coordinate system with constant width.
    The width of the rings and the inner radius of the biggest ring can be
    changed by setting the optional parameters ring_width and max_radius,
    respectively.
    .. note:: To calculate the radially averaged power density spectra
              use the outputs of the function power_density_spectra as
              input of this one.
    Parameters:
    * kx, ky : 2D-arrays
        The wavenumbers arrays in the `x` and `y` directions
    * pds : 2D-array
        The Power Density Spectra
    * max_radius : float (optional)
        Inner radius of the biggest ring.
        By default it's set as the minimum of kx.max() and ky.max().
        Making it smaller leaves points outside of the averaging,
        and making it bigger includes points nearer to the boundaries.
    * ring_width : float (optional)
        Width of the rings.
        By default it's set as the largest value of :math:`\Delta k_x` and
        :math:`\Delta k_y`, being them the equidistances of the kx and ky
        arrays.
        Making it bigger gives more populated averages, and
        making it smaller lowers the ammount of points per ring
        (use it carefully).
    Returns:
    * k_radial : 1D-array
        Wavenumbers of each Radially Averaged Power Spectrum point.
        Also, the inner radius of the rings.
    * pds_radial : 1D array
        Radially Averaged Power Spectrum
    """
    nx, ny = pds.shape
    if max_radius is None:
        max_radius = min(kx.max(), ky.max())
    if ring_width is None:
        ring_width = max(numpy.unique(kx)[numpy.unique(kx) > 0][0],
                         numpy.unique(ky)[numpy.unique(ky) > 0][0])
    k = numpy.sqrt(kx**2 + ky**2)
    pds_radial = []
    k_radial = []
    radius_i = -1
    while True:
        radius_i += 1
        if radius_i*ring_width > max_radius:
            break
        else:
            if radius_i == 0:
                inside = k <= 0.5*ring_width
            else:
                inside = numpy.logical_and(k > (radius_i - 0.5)*ring_width,
                                           k <= (radius_i + 0.5)*ring_width)
            pds_radial.append(pds[inside].mean())
            k_radial.append(radius_i*ring_width)
    return numpy.array(k_radial), numpy.array(pds_radial)


def to_radial(kx, ky, psd2d):
    a = np.fliplr(psd2d[:, 1:-1])
    psd2d = np.concatenate((psd2d, a), axis=1)
    ky = np.concatenate((ky, np.flip(ky[:, 1:-1])), axis=1)
    kx = np.concatenate((kx, np.flip(kx[:, 1:-1])), axis=1)
    return radial_average_spectrum(kx, ky, psd2d)

def energy_budget(m, path_output_dir: str = None, size: int = None):
    # some spectral plots
    KE1spec = m.get_diagnostic('KEspec')[0].sum(
        axis=0)  # note that this is misleading for anisotrphic flows...
    KE2spec = m.get_diagnostic('KEspec')[1].sum(
        axis=0)  # we should sum azimuthaly, and plot as a functions of kappa
    try:
        param_spec = m.get_diagnostic('ADVECparam')[0].sum(axis=0)
    except:
        print('Could not find the parameterization diagnostic')

    # factor ebud
    ebud_factor = 1.e4
    ebud_factor_s = str('%1.1e') % ebud_factor

    # inertial range
    ir = np.r_[10:20]

    fig = plt.figure(figsize=(16., 7.))
    ax1 = fig.add_subplot(121)
    ax1.loglog(m.kk, KE1spec, '.-')
    ax1.loglog(m.kk, KE2spec, '.-')
    try:
        ax1.loglog(m.kk, param_spec, '-*')
    except:
        print('Spectrum of parameterization not defined')
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
    ebud = [m.get_diagnostic('APEgenspec'),
            m.get_diagnostic('APEflux'),
            m.get_diagnostic('KEflux'),
            m.get_diagnostic('adv_param'),
            -m.rek * m.del2 * m.get_diagnostic('KEspec')[1] * m.M ** 2]
    ebud.append(-np.stack(ebud).sum(axis=0))
    ebud_labels = ['APE gen', 'APE flux', 'KE flux', 'parameterization',
                   'Bottom drag diss.' , 'Resid.']

    ax2 = fig.add_subplot(122)
    [ax2.semilogx(*(to_radial(m.k, m.l, term))) for term in ebud]
    ax2.legend(ebud_labels, loc='upper right')

    kk, _ = to_radial(m.k, m.l, ebud[0])
    ax2.grid()
    ax2.set_xlim([kk.min(), kk.max()])
    ax1.set_xlabel(r'k (m$^{-1})$')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.set_title(r'      Spectral Energy Budget')

    ax2.set_xlabel(r'k (m$^{-1})$')

    if path_output_dir:
        plt.savefig(path_output_dir / f'energy_budget{size}.jpg', dpi=600)

    # New figure
    plt.figure()
    n_plots = len(ebud)
    for i in range(n_plots):
        plt.subplot(1, n_plots, i + 1)
        plt.imshow(fftshift(ebud[i], axes=(0,)))
        plt.title(ebud_labels[i])
        plt.colorbar()

    if path_output_dir:
        plt.savefig(path_output_dir / f'energy_budget2D{size}.jpg', dpi=600)

def play_movie(predictions: np.ndarray, title: str = '',
               interval: int = 500):
    fig = plt.figure()
    ims = list()
    mean = np.mean(predictions)
    std = np.std(predictions)
    vmin, vmax = mean - std, mean + std
    for im in predictions:
        ims.append([plt.imshow(im, vmin=vmin, vmax=vmax,
                               cmap='YlOrRd',
                               origin='lower', animated=True)])
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=1000)
    plt.title(title)
    plt.show()
    return ani