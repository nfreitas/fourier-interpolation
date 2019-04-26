import numpy as np
import imageio

def test_data():

    im = imageio.imread('./image-710x473.jpg')

    return np.array(im[:,:,0], dtype=complex)

def fi(A, factor):
    """Regular Fourier interpolation. Only used to test large_fi."""

    N, M = A.shape # I assume even N and M

    F = np.fft.fft2(A)

    G = np.zeros((factor*N, factor*M),dtype=complex)
    G[:int(N/2),:int(M/2)] = F[:int(N/2),:int(M/2)]
    G[:int(N/2),-int(M/2):] = F[:int(N/2),-int(M/2):]
    G[-int(N/2):,:int(M/2)] = F[-int(N/2):,:int(M/2)]
    G[-int(N/2):,-int(M/2):] = F[-int(N/2):,-int(M/2):]

    # I am ignoring the normalization factor by using fft2 to actually
    # compute ifft2
    B = np.conj(np.fft.fft2(np.conj(G)))
    # I normalize by hand
    B = B/(N*M)

    return F,B

def large_fi(A, factor, band):
    """Fourier interpolation for large matrices.

    A: The data to interpolate. It must be a NxM matrix with even N and M

    factor: The scaling factor such that the full output would be of size (factor*N)x(factor*M)

    band: An array with the indexes required over the second axis. If the length of
    this array is K, then the output is of size (factor*N)xK.
    """

    N, M = A.shape # I assume even N and M

    F = np.fft.fft2(A)

    ANK, OUT = band_padding_fft2d(np.conj(F), factor, band)
    OUT = np.conj(OUT)/(N*M)

    return ANK, OUT

def band_padding_fft2d(A, factor, band):
    """A simple implementation of fft2d that automatically takes into account
    large-momentum-zero-padding and only computes a required band or slice of the output"""

    N, M = A.shape # I assume even N and M

    # compute the partial transform a_n(k)
    indM = np.zeros(M).reshape(M,1)
    indM[:int(M/2),0] = np.arange(int(M/2))
    indM[-int(M/2):,0] = np.arange(int((factor-1/2)*M), factor*M)

    indK = np.array(band).reshape(1,len(band))

    indMK = indM.dot(indK)

    complex_factors = np.exp(-1j*2*np.pi*indMK/M)

    ANK = A.dot(complex_factors)

    # now compute one ftt1d for each column of the band
    OUT = np.fft.fft(ANK,axis=0)

    return OUT
