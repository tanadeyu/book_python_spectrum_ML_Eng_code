import numpy as np
import matplotlib.pyplot as plt

# Sampling frequency [Hz]
fs = 1000

# Create time course array
t = np.arange(0, 1, 1/fs)

# Generates three cosine waves
f = 100
x1 = 1*np.cos(2*np.pi*f*t)
x2 = 0.75*np.cos(2*np.pi*f*t + np.pi/2)
x3 = 1.5*np.cos(2*np.pi*f*t - np.pi/2)

# Perform Fourier transform on each
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)
X3 = np.fft.fft(x3)

# Calculation of magnitude spectrum
amp1 = np.abs(X1)
amp2 = np.abs(X2)
amp3 = np.abs(X3)

# Calculate amplitude value by scaling
amp1_norm = amp1 / fs *2
amp2_norm = amp2 / fs *2
amp3_norm = amp3 / fs *2

# Phase spectrum calculation
phase1 = np.angle(X1, deg=False)
phase2 = np.angle(X2, deg=False)
phase3 = np.angle(X3, deg=False)

# Phase information is shown in the range of -180 degrees to 180 degrees.
phase1 = np.arctan2(np.sin(phase1), np.cos(phase1))/np.pi*180
phase2 = np.arctan2(np.sin(phase2), np.cos(phase2))/np.pi*180
phase3 = np.arctan2(np.sin(phase3), np.cos(phase3))/np.pi*180

# Set the graph format
fig, axs = plt.subplots(4, 3, figsize=(7, 7))

# Original data
axs[0, 0].plot(t, x1)
axs[0, 0].set_title('Data 1')
axs[0, 0].set_xlim(0.0,0.1)
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Amplitude')
axs[0, 1].plot(t, x2)
axs[0, 1].set_title('Data 2')
axs[0, 1].set_xlim(0.0,0.1)
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 2].plot(t, x3)
axs[0, 2].set_title('Data 3')
axs[0, 2].set_xlim(0.0,0.1)
axs[0, 2].set_xlabel('Time [s]')
axs[0, 2].set_ylabel('Amplitude')

# Magnitude spectrum
axs[1, 0].plot(amp1_norm)
axs[1, 0].set_xlabel('Frequency [Hz]')
axs[1, 0].set_ylabel('Magnitude')
axs[1, 0].set_xlim(0,500)
axs[1, 1].plot(amp2_norm)
axs[1, 1].set_xlabel('Frequency [Hz]')
axs[1, 1].set_ylabel('Magnitude')
axs[1, 1].set_xlim(0,500)
axs[1, 2].plot(amp3_norm)
axs[1, 2].set_xlabel('Frequency [Hz]')
axs[1, 2].set_ylabel('Magnitude')
axs[1, 2].set_xlim(0,500)

# Display Phase
axs[2, 0].plot(phase1)
axs[2, 0].set_xlabel('Frequency [Hz]')
axs[2, 0].set_ylabel('Phase[deg]')
axs[2, 0].set_xlim(0,500)
axs[2, 1].plot(phase2)
axs[2, 1].set_xlabel('Frequency [Hz]')
axs[2, 1].set_ylabel('Phase[deg]')
axs[2, 1].set_xlim(0,500)
axs[2, 2].plot(phase3)
axs[2, 2].set_xlabel('Frequency [Hz]')
axs[2, 2].set_ylabel('Phase[deg]')
axs[2, 2].set_xlim(0,500)

# Display Phase in a limited range
axs[3, 0].plot(phase1)
axs[3, 0].set_xlabel('Frequency [Hz]')
axs[3, 0].set_ylabel('Phase[deg]')
axs[3, 0].set_xlim(98,102)
axs[3, 0].set_ylim(-180,180)
axs[3, 1].plot(phase2)
axs[3, 1].set_xlabel('Frequency [Hz]')
axs[3, 1].set_ylabel('Phase[deg]')
axs[3, 1].set_xlim(98,102)
axs[3, 1].set_ylim(-180,180)
axs[3, 2].plot(phase3)
axs[3, 2].set_xlabel('Frequency [Hz]')
axs[3, 2].set_ylabel('Phase[deg]')
axs[3, 2].set_xlim(98,102)
axs[3, 2].set_ylim(-180,180)

# Phase at 100Hz
print('phase1@100Hz[degree] =',phase1[100])
print('phase2@100Hz[degree] =',phase2[100])
print('phase3@100Hz[degree] =',phase3[100])

# Visualize the graph
plt.tight_layout()
plt.show()
