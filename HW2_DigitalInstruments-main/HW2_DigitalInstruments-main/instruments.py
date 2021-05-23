import numpy as np
import matplotlib.pyplot as plt


def load_tune(filename, tune_length):
    """
    Load in information about notes and their
    onset times from a text file
    Parameters
    ----------
    filename: string
        Path to file with the tune
    tune_length: float
        Length, in seconds, of the tune
    
    Returns
    -------
    ps: ndarray(N)
        A list of N note numbers
    times: ndarray(N)
        Duration of each note, in increments
        of sixteenth notes
    """
    tune = np.loadtxt(filename)
    ps = tune[:, 0]
    times = np.zeros(tune.shape[0])
    times[1::] = np.cumsum(tune[0:-1, 1])
    times = times*tune_length/np.sum(tune[:, 1])
    return ps, times

def karplus_strong_note(sr, note, duration, decay):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    decay: float 
        Decay amount (between 0 and 1)

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    N = int(duration*sr)
    y = np.zeros(N)
    f = 440*2**(note/12) #stolen from assignment 1, the eq is 440*2**(note/12)
    T = int(sr/f)#this should calculate the period in samples per f seconds
    # we want to initialize the first T samples as random noise and populate 0 through T with random values
    y[0:T] = np.random.rand(T) #generates T random noise samples
    #next we populate the y array
    for i in range(T, N):
        y[i] = decay*((y[i-T]+y[i-T+1])/2)
    return y

def fm_synth_note(sr, note, duration, ratio = 2, I = 2, 
                  envelope = lambda N, sr: np.ones(N),
                  amplitude = lambda N, sr: np.ones(N)):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    ratio: float
        Ratio of modulation frequency to carrier frequency
    I: float
        Modulation index (ratio of peak frequency deviation to
        modulation frequency)
    envelope: function (N, sr) -> ndarray(N)
        A function for generating an ADSR profile
    amplitude: function (N, sr) -> ndarray(N)
        A function for generating a time-varying amplitude

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    N = int(duration*sr)
    y = np.zeros(N)
    t = np.arange(N)/sr
    fc = 440*2**(note/12)
    fm = fc*ratio
    env= I*envelope(N, sr)  #numpy array that is the full time variant
    amp = amplitude(N, sr)
    #y(t) = A(t)cos(2*pi*fc*t+I(t)sin(2*pi*fm*t))
    y = amp*np.cos(2*np.pi*fc*t + env*np.sin(2*np.pi*fm*t))
    return y

def exp_env(N, sr, lam = 3):
    """
    Make an exponential envelope
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    lam: float
        Exponential decay rate: e^{-lam*t}

    Returns
    -------
    ndarray(N): Envelope samples
    """
    return np.exp(-lam*np.arange(N)/sr)

def drum_like_env(N, sr):
    """
    Make a drum-like envelope, according to Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    t = np.arange(N)/sr
    y = np.zeros(N)
    y = ((t+.025)**2)*np.exp(-12*(t+.025)*3)
    return y

def wood_drum_env(N, sr):
    """
    Make the wood-drum envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    y = np.zeros(N)
    y[0:int(sr*0.025)] = np.linspace(1, 0, int(sr*0.025))
    return y

def dirty_bass_env(N, sr):
    """
    Make the "dirty bass" envelope from Attack Magazine
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    y = np.zeros(N)
    y[0:int(.25*sr)] = np.exp(-29*np.arange(int(N/2))/sr)
    y[int(.25*sr):int(.5*sr)] = 1-np.exp(-29*np.arange(int(N/2))/sr)
    return y

def fm_plucked_string_note(sr, note, duration, lam = 3):
    """
    Make a plucked string of a particular length
    using FM synthesis
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    lam: float
        The decay rate of the note
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, lam)
    return fm_synth_note(sr, note, duration, \
                ratio = 1, I = 8, envelope = envelope,
                amplitude = envelope)


def brass_env(N, sr):
    """
    Make the brass ADSR envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    #N = dur*sr
    #dur = N/sr
    y = np.zeros(N)
    dur = N/sr
    if dur <= 0.3:
        atk = np.linspace(0, 1, int(sr*0.05))
        dec = np.linspace(1, 0.75, (int(sr*0.2)-int(sr*0.05)))
        sus = np.linspace(0.75, 0.75, (int(sr*0.95) - int(sr*0.05)))
        rel = np.linspace(0.75, 0, (int(sr*1) - int(sr*0.95)))
        y = np.concatenate((atk, dec, sus, rel))
        
    else:
        atk = np.linspace(0, 1, int(sr*0.1))
        dec = np.linspace(1, 0.75, int(sr*0.1))
        sus = np.linspace(0.75, 0.7, (int(sr*0.9)-int(sr*0.2)))
        rel = np.linspace(0.7, 0, int(sr*0.1))
        y = np.concatenate((atk, dec, sus, rel))
    return y


def fm_bell_note(sr, note, duration):
    """
    Make a bell note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, 0.8)
    return fm_synth_note(sr, note, duration, \
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope)

def fm_brass_note(sr, note, duration):
    """
    Make a brass note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    
    envelope = lambda N, sr: brass_env(N, sr)
    return fm_synth_note(int(sr), note, duration, \
                ratio = 1, I = 10, envelope = envelope,
                amplitude = envelope)

def fm_drum_sound(sr, note, duration, fixed_note = -14):
    """
    Make what Chowning calls a "drum-like sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    ------
    ndarray(N): Audio samples for this drum hit
    """
    envelope = lambda N, sr: drum_like_env(N, sr)
    return fm_synth_note(sr, note, duration, \
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope)

def fm_wood_drum_sound(sr, note, duration, fixed_note = -14):
    """
    Make what Chowning calls a "wood drum sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    envelope = lambda N, sr: wood_drum_env(N, sr)
    return fm_synth_note(sr, note, duration, \
                ratio = 1.4, I = 10, envelope = envelope,
                amplitude = envelope)

def fm_dirty_bass_note(sr, note, duration):
    """
    Make a "dirty bass" note, based on 
    https://www.attackmagazine.com/technique/tutorials/dirty-fm-bass/
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    
    envelope = lambda N, sr: dirty_bass_env(N, sr)
    return fm_synth_note(sr, note, duration, \
                ratio = 1, I = 18, envelope = envelope,
                amplitude = envelope) 

def make_tune(filename, sixteenth_len, sr, note_fn):
    """
    Parameters
    ----------
    filename: string
        Path to file containing the tune.  Consists of
        rows of <note number> <note duration>, where
        the note number 0 is a 440hz concert A, and the
        note duration is in factors of 16th notes
    sixteenth_len: float
        Length of a sixteenth note, in seconds
    sr: int
        Sample rate
    note_fn: function (sr, note, duration) -> ndarray(M)
        A function that generates audio samples for a particular
        note at a given sample rate and duration
    
       
    Returns
    -------
    -------
    ndarray(N): Audio containing the tune
    """
    tune = np.loadtxt(filename)
    notes = tune[:, 0]
    durations = tune[:, 1]
    durations_sec = durations*sixteenth_len
    ret = []
    n = len(notes) #number of notes
    for i in range(n):
        x = note_fn(sr, notes[i], durations_sec[i])
        if np.isnan(notes[i]): #if the note is not a number, append 0 into y such that it has no value i.e is silent
            x = np.zeros(int(durations_sec[i]*sr))
            ret = np.concatenate((ret, x))
        else:
            ret = np.concatenate((ret, x))
    return ret 