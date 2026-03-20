import numpy as np
import control as ctl
import matplotlib.pyplot as plt

def generate_stim(fs, duration, freq=30, pulse_width=0.001):
    t = np.arange(0, duration, 1/fs)
    stim = np.zeros_like(t)

    period = int(fs / freq)
    width = int(pulse_width * fs)
    half = width // 2

    for k in range(0, len(t), period):
        if k + width < len(t):
            stim[k:k+half] = 1
            stim[k+half:k+width] = -1

    return stim


def artifact_kernel(fs):
    t = np.arange(0, 0.02, 1/fs)  # 20 ms kernel

    # Fast electrode discharge
    tau1, tau2 = 0.0015, 0.0003
    fast = np.exp(-t/tau1) - np.exp(-t/tau2)

    # Ringing (hardware resonance)
    fr = 180 + 40*np.random.randn()   # variability
    tau3 = 0.003
    ringing = np.exp(-t/tau3) * np.sin(2*np.pi*fr*t)

    # Slow drift / polarization
    tau4 = 0.02
    slow = np.exp(-t/tau4)

    return 1.5*fast + 0.3*ringing + 0.1*slow

def generate_artifact(stim, fs):
    edges = np.where(np.abs(np.diff(stim)) > 0)[0]
    signal = np.zeros_like(stim)

    for j, idx in enumerate(edges):
        factor = 1 if j % 2==0 else -1
        # Random amplitude (impedance variability)
        A = 1 + 0.2*np.random.randn()

        # Slight timing jitter
        jitter = int(np.random.normal(0, 0.0002*fs))
        i = idx + jitter
        if i < 0 or i >= len(signal):
            continue

        # Slight kernel variation
        h = artifact_kernel(fs) * (A * factor)

        L = len(h)
        if i + L < len(signal):
            sign = np.sign(stim[idx+1] - stim[idx])
            signal[i:i+L] += sign * h

    return signal
stim = generate_stim(50000, 1, freq=30, pulse_width=0.001)
art = generate_artifact(stim, 50000)
    
plt.plot(stim, label='square wave')
plt.show(block=True)
plt.plot(art, label='artifact')
plt.show(block=True)