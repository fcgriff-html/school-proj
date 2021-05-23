import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
from scipy import sparse
from scipy.io import wavfile

def load_wavfile(filename, mono=True):
    """
    Load a wav file using scipy's wavfile input
    
    Parameters
    ----------
    filename: string
        Path to wave file
    mono: boolean
        Whether to mix stereo audio to mono
    
    Returns
    -------
    y: ndarray(N, dtype=np.float32)
        Audio samples in the range [-1, 1]
    sr: int
        Audio sample rate
    """
    sr, y = wavfile.read(filename)
    if y.dtype == np.uint8:
        y = 2*(y/255 - 0.5)
    elif y.dtype == np.int16:
        y = y/32768
    elif y.dtype == np.int32:
        y = y/2147483648
    if mono and len(y.shape) > 1:
        y = np.mean(y, axis=1)
    return y, sr

def get_oti(C1, C2, do_plot = False):
    """
    Compute the optimal transposition index between two chroma
    sequences

    Parameters
    ----------
    C1: ndarray(12, M)
        First chromagram
    C2: ndarray(12, N)
        Second chromagram
    
    Returns
    -------
    int: The optimal shift bringing C1 into alignment with C2
    """
    # Average all chroma window into one representative chroma window
    # for each tune
    gC1 = np.mean(C1, axis=1)
    gC2 = np.mean(C2, axis=1)
    corr = np.zeros(len(gC1))
    ## TODO: Fill this in.  Make corr[i] the distance between gC1
    ## circularly shifted by i and gC2, for all i between 0 and 11
    dist = []
    for i in range(0, 11):
        shift_gC1 = np.roll(gC1, i)
        corr[i] = np.root((gC2[i]-gC1[i])**2)
    if do_plot:
        plt.plot(corr)
        plt.xticks(np.arange(12))
        plt.xlabel("Halfstep Shift of First Version")
        plt.ylabel("Global Distance")
        plt.title("Global Transposition Scores")
    return np.argmin(corr)

def get_csm_oti(C1, C2, oti):
    """
    Compute the euclidean cross-similarity between two chroma
    vectors after circularly shifting the first one, using
    numpy broadcasting for speed

    Parameters
    ----------
    C1: ndarray(12, M)
        First chromagram
    C2: ndarray(12, N)
        Second chromagram
    oti: int
        The amount by which to circularly shift all windows of the
        first chromagram before computing the CSM
    
    Returns
    -------
    ndarray(M, N)
        The cross-similarity between all chroma windows in C1 versus
        all chroma windows in C2
    """
    Y1 = np.roll(C1, oti, axis=0).T
    Y2 = C2.T
    Y1Sqr = np.sum(Y1**2, 1)
    Y2Sqr = np.sum(Y2**2, 1)
    D = Y1Sqr[:, None] + Y2Sqr[None, :] - 2*Y1.dot(Y2.T)
    D[D < 0] = 0
    D = np.sqrt(D)
    return D

def get_csm_enhanced(C1, C2, b):
    """
    Perform the effect of a sliding window on an CSM by averaging
    along diagonals

    Parameters
    ----------
    C1: ndarray(12, M)
        First chromagram
    C2: ndarray(12, N)
        Second chromagram
    b: int
        Length of block in which to average diagonal entries
    
    Returns
    -------
    ndarray(M-b+1, N-b+1): The diagonally averaged CSM
    """
    M = C1.shape[1] - b + 1
    N = C2.shape[1] - b + 1
    Db = np.zeros((M, N))
    oti = get_oti(C1, C2)
    csm = get_csm_oti(C1, C2, oti)
    summy = 0
    for i in range(M):
        for j in range(N):
            for k in range(b):
                summy += csm[i+k, j+k]
            Db[i, j] = (1/b)*summy
    ## TODO: Fill this in.  Compute the OTI and the original CSM.
    ## Then, diagonally average the CSM and store the result in S
    return Db

def binary_csm(CSM, k):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix, where 
    an entry (i, j) is a 1 if and only if it is within the both the nearest
    neighbor set of i and the nearest neighbor set of j
    Parameters
    ----------
    CSM: ndarray(M, N)
        M x N cross-similarity matrix
    k: int
        The number of mutual neighbors to consider
    
    Returns
    -------
    B: ndarray(M, N)
        MxN binary cross-similarity matrix
    """
    ## TODO: Fill this in
    #fix number k
    #for i and j in csm, B[i,j] = 1 if B[i,j] is similar to k??
    for i in range(CSM.shape[0]):
        part_i = np.partition(CSM[i, :], k)[k]
        for j in range(CSM.shape[1]):
            part_j = np.partition(CSM[:, j], k)[k]
            if part_i > CSM[i, j] and part_j > CSM[i, j]:
                B[i, j] = 1
            else:
                B[i, j] = 0
    return B
def gamma(CRP, i, j):
    x=0
    if CRP[i, j] == 1:
        x = 1
    else: 
        x = 0.5
    return x

def qmax(CRP):
    """
    Compute the qmax table Q from a given cross-recurrence plot

    Parameters
    ----------
    CRP: ndarray(M, N)
        A binary cross-similarity matrix between two tunes
    
    Returns
    ndarray(M, N)
        The qmax matrix Q extracted from the CRP
    """
    M = CRP.shape[0]
    N = CRP.shape[1]
    Q = np.zeros_like(CRP)
    if (N < 3 or M < 3):
        return Q
    
    for i in range(M):
        for j in range(N):
            if CRP[i, j] == 1:
                Q[i, j] = max(Q[i-1, j-1], Q[i-2, j-1], Q[i-1, j-2]) + 1
            else: 
                Q[i, j] = max(0, (Q[i-1, j-1] - gamma(CRP, i-1, j-1)), (Q[i-2, j-1] - gamma(CRP, i-2, j-1)), (Q[i-1, j-2] - gamma(CRP, i-1, j-2))
                

    return Q
