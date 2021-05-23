"""
Purpose: To implementing the NMF techniques in [1]
[1] Driedger, Jonathan, Thomas Praetzlich, and Meinard Mueller. 
"Let it Bee-Towards NMF-Inspired Audio Mosaicing." ISMIR. 2015.
"""
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import time
import librosa


def create_musaic(S, WComplex, win_length, hop_length, L, r=3, p=10, c=3):
    """
    Implement the technique from "Let It Bee-Towards NMF-Inspired
    Audio Mosaicing"

    Parameters
    ----------
    S: ndarray(M, N, dtype=np.complex)
        A M x N nonnegative target matrix
    WComplex: ndarray(M, K, dtype=np.complex) 
        An M x K matrix of template sounds in some time order along the second axis
    win_length: int
        Window length of STFT (used in Griffin Lim)
    hop_length: int
        Hop length of STFT (used in Griffin Lim)
    L: int
        Number of iterations
    r: int
        Half of the width of the repeated activation filter
    p: int
        Degree of polyphony; i.e. number of values in each column of H which should be 
        un-shrunken
    c: int
        Half length of time-continuous activation filter
    """
    V = np.abs(S) # V is the absolute magnitude spectrogram, keeping it nonnegative #v is basically sAbS
    W = np.abs(WComplex) # W is the absolute magnitude spectrogram of WComplex
    N = V.shape[1]
    K = W.shape[1]
    WDenom = np.sum(W, 0)
    WDenom[WDenom == 0] = 1
    
    # Random nonnegative initialization of H
    H = np.random.rand(K, N)
    for l in range(L):
        print(l, end='.') # Print out iteration number for progress
         #above is the griffin-lim stuff
        # Step 1: Avoid repeated activations
        for i in range(K):
            for j in range(N):
                if r<= j:
                    if H[i, j] != max(H[i, j-r:j+r]):
                        H[i, j] *= (1-(l/L))*H[i, j]    
    
        # Step 2: Restrict number of simultaneous activations
               
        for i in range(K):
            for j in range(N):
                part = np.partition(H[i, :], p)[p]
                if H[i, j] < part:
                    H /= 1-(l/L)
        
        # Step 3: Supporting time-continuous activations
        h = np.zeros_like(H)
        for i in range(K):
            for j in range(N):
                for k in range(-c, c+1):
                    if i+k < K and j+k < N:
                        h[i, j] += H[i+k, j+k]
    
        # Step 4: Match target with an iteration of KL-based NMF, keeping
        # W fixed
        WH = W.dot(H)
        WH[WH == 0] = 1 # Prevent divide by 0
        VLam = V/WH
        H = H*((W.T).dot(VLam)/WDenom[:, None])
    
    #y = librosa.istft(WComplex.dot(H), win_length=win_length, hop_length=hop_length)
    ## TODO: Use 10 iterations of Griffin-Lim instead of a straight-up STFT - done
    #S = WComplex.dot(H)
    #V = sABS
    for i in range(L):    
        sABS = np.abs(WComplex.dot(H))
        A = librosa.stft(librosa.istft(WComplex.dot(H), win_length = win_length, hop_length = hop_length), win_length = win_length, hop_length = hop_length)
        phase = np.arctan2(np.imag(A), np.real(A)) 
        eip = np.exp(np.complex(0, 1)*phase) 
        S = eip*sABS
    y = np.real(librosa.istft(S, win_length = win_length, hop_length = hop_length))
    
    return y
