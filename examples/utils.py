import matplotlib.pyplot as plt
import numpy as np
import numpy
from pyqg.model import Model
from matplotlib import animation
from scipy.fft import fftshift
from scipy.ndimage import gaussian_filter
from numpy.fft import fft2, fftshift, fftn, ifft2, fftfreq

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
    radius_i = 0
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


def energy_budget(m, path_output_dir: str = None, size: int = None,
                  ymin = None, ymax=None):
    # some spectral plots
    KE1spec = m.get_diagnostic('KEspec')[0]
    KE2spec = m.get_diagnostic('KEspec')[1]
    # k = m.kk
    # KE1spec = KE1spec.sum(axis=0)
    # KE2spec = KE2spec.sum(axis=0)
    k, KE1spec = to_radial(m.k, m.l, KE1spec)
    k, KE2spec = to_radial(m.k, m.l, KE2spec)
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
    ax1.loglog(k, KE1spec, '.-')
    ax1.loglog(k, KE2spec, '.-')
    try:
        ax1.loglog(m.kk, param_spec, '-*')
    except:
        print('Spectrum of parameterization not defined')
    ax1.loglog(k[10:20], 2 * (k[ir] ** -3) *
               KE1spec[ir].mean() / (k[ir] ** -3).mean(),
               '0.5')
    ax1.set_ylim([1e-9, 1e-3])
    ax1.set_xlim([k.min(), k.max()])
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
    ax2.set_ylim([ymin, ymax])
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
               interval: int = 500, vmin=None, vmax=None):
    fig = plt.figure()
    ims = list()
    mean = np.mean(predictions)
    std = np.std(predictions)
    if vmin is None:
        vmin, vmax = mean - 2 * std, mean + 2 * std
    for im in predictions:
        ims.append([plt.imshow(im, vmin=vmin, vmax=vmax,
                               cmap='YlOrRd',
                               origin='lower', animated=True)])
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=1000)
    plt.title(title)
    plt.show()
    return ani


def coarsen(data, factor: int = 4, axes=(-1, -2)):
    scales = [0] * data.ndim
    for i in axes:
        scales[i] = factor / 2.
    filtered = gaussian_filter(data, scales)
    result = np.zeros((data.shape[0:2]) + (data.shape[2] // factor,
                                           data.shape[3] // factor))
    for i in range(factor):
        for j in range(factor):
            result += filtered[..., i::factor, j::factor]
    return result / factor**2


def fft_coarsen(data, factor: int = 4, axes=(-1, -2)):
    spec = fft2(data)
    k_x = fftfreq(spec.shape[-1])
    k_y = fftfreq(spec.shape[-2])
    k_x, k_y = np.meshgrid(k_x, k_y, indexing='ij')
    sel = np.maximum(abs(k_x), abs(k_y)) > (1/2/factor)
    print(sel.shape)
    sel = sel.reshape((1, 1, sel.shape[0], sel.shape[1]))
    sel = sel.repeat(spec.shape[1], axis=1)
    sel = sel.repeat(spec.shape[0], axis=0)
    print(sel.shape)
    spec[sel] = 0
    result = np.zeros((data.shape[0:2]) + (data.shape[2] // factor,
                                           data.shape[3] // factor))
    data = np.real(ifft2(spec))
    for i in range(factor):
        for j in range(factor):
            result += data[..., i::factor, j::factor]
    return result / factor**2

def spatial_spectrum(uv, radial: bool = True, co: bool = False,
                     L: float = 1.2e3):
    if uv.ndim == 3:
        uv = uv.reshape((1,) + uv.shape)
    uv = uv[:, 0, ...] + 1j * uv[:, 1, ...]
    dx = L / uv.shape[-1]
    dy = L / uv.shape[-2]
    print('dx', dx)
    if not co:
        spectrum_2d = dx * dy / (uv.shape[-1] * uv.shape[-2]) * abs(fft2(uv))**2
    else:
        spectrum_2d = dx * dy / (uv.shape[-1] * uv.shape[-2]) * abs(fft2(uv)**2)
    # Time-average of spatial spectra
    spectrum_2d = spectrum_2d.mean(axis=0)
    if not radial:
        return spectrum_2d
    k_x = fftfreq(spectrum_2d.shape[0]) / dx
    k_y = fftfreq(spectrum_2d.shape[1]) / dy
    k_x, k_y = np.meshgrid(k_x, k_y, indexing='ij')
    print('max', np.max(k_x))
    return radial_average_spectrum(k_x, k_y, spectrum_2d)



def spatio_temporal_spectrum(uv):
    uv = uv[:, 0, ...] + 1j * uv[:, 1, ...]
    shape = uv.shape
    return 1 / (shape[0] * shape[1] * shape[2])**2 * abs(fftn(uv))**2

def _initialize_filter(m, n):
    """Set up frictional filter."""
    # this defines the spectral filter (following Arbic and Flierl, 2003)
    cphi=0.65*np.pi
    k = fftfreq(m) * 2 * np.pi
    l = fftfreq(n) * 2 * np.pi
    k, l = np.meshgrid(k, l, indexing='ij')
    wvx=np.sqrt(k**2. + l**2)
    filtr = np.exp(-23.6*(wvx-cphi)**4.)
    filtr[wvx<=cphi] = 1.
    return filtr


def same_freq_grid(spec1, spec2):
    shape1 = spec1.shape
    shape2 = spec2.shape
    if shape1[-1] > shape2[-1]:
        spec2, spec1 = same_freq_grid(spec2, spec1)
        return spec1, spec2
    block1 = spec2[..., :shape1[-2] // 2 + 1, :shape1[-1] // 2 + 1]
    block2 = spec2[..., -shape1[-2] // 2 + 1:, :shape1[-1] // 2 + 1]
    block3 = spec2[..., -shape1[-2] // 2 + 1:, -shape1[-1] // 2 + 1:]
    block4 = spec2[..., :shape1[-2] // 2 + 1, -shape1[-1] // 2 + 1:]
    block1 = np.concatenate((block1, block4), axis=-1)
    block2 = np.concatenate((block2, block3), axis=-1)
    result = np.concatenate((block1, block2), axis=-2)
    return spec1, result


def kullback_leibler(uv1, uv2, temporal=True):
    spec1 = spatio_temporal_spectrum(uv1)
    spec2 = spatio_temporal_spectrum(uv2)
    spec1, spec2 = same_freq_grid(spec1, spec2)
    if not temporal:
        spec1 = np.mean(spec1, axis=0, keepdims=True)
        spec2 = np.mean(spec2, axis=0, keepdims=True)
    shape = spec1.shape
    spec1 = fftshift(spec1)
    spec2 = fftshift(spec2)
    filter = fftshift(_initialize_filter(shape[1], shape[2]))
    # spec1 *= filter
    freq_x = fftfreq(shape[1]) * shape[1]
    freq_y = fftfreq(shape[2]) * shape[2]
    ks = np.meshgrid(freq_x, freq_y, indexing='ij')
    sel = np.logical_and((ks[0]**2 + ks[1]**2) <= (shape[1] * 0.65 / 2)**2,
                         (ks[0]**2 +ks[1]**2) >=0)
    sel = fftshift(sel)
    sel = sel.reshape((1, sel.shape[0], sel.shape[1]))
    sel = sel.repeat(shape[0], axis=0)
    term1 = spec1 / spec2
    term2 = - np.log(spec1 / spec2)
    kl_div = term1 + term2 - 1
    term1 = np.where(sel, term1, np.nan)
    term2 = np.where(sel, term2, np.nan)
    kl_div = np.where(sel, kl_div, np.nan)
    filter = filter.reshape((1, filter.shape[0], filter.shape[1]))
    kl_div *= filter
    return np.nanmean(kl_div)


def j_divergence(uv1, uv2, *args):
    return kullback_leibler(uv1, uv2, *args) + kullback_leibler(uv2, uv1, *args)
