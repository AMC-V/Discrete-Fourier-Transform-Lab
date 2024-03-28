from scipy.io import wavfile
from IPython.display import Audio

import numpy as np
import matplotlib.pyplot as plt

#!wget https://upload.wikimedia.org/wikipedia/commons/c/c3/Dominant_seventh_chord.wav

from IPython.display import clear_output

clear_output()

print(".wav file loaded")

data = wavfile.read('Dominant_seventh_chord.wav')

# Separete the object elements
samplerate = data[0]
sounddata = data[1]
time      = np.arange(0,len(sounddata))/samplerate

# Show information about the object
print('Sample rate:',samplerate,'Hz')
print('Total time:',len(sounddata)/samplerate,'s')

Audio(sounddata.T,rate=samplerate) #<---Note not all sound files will need to be transposed with a .T like this, but this one does.

dft_sounddata = np.fft.fft(sounddata[:,0])
freq = np.fft.fftfreq(n=sounddata.shape[0],d=1./44100)

plt.subplots_adjust(hspace=1.5)
fig, axs = plt.subplots(3, 1, constrained_layout=False)
axs[0].plot(time,sounddata[:,0])
#axs[0].set_title('subplot 1')
axs[0].set_xlabel('time')
axs[0].set_ylabel('signal')
fig.suptitle('Example Sound Output', fontsize=16)

axs[1].plot(freq,dft_sounddata.real)
axs[1].set_xlim(-1500, 1500) #<---- These limits can be used to zoom in on the spectrum
axs[1].set_xlabel('frequency')
#axs[1].set_title('real')
axs[1].set_ylabel('Real')

axs[2].plot(freq,dft_sounddata.imag)
axs[2].set_xlim(-1500,1500)
axs[2].set_xlabel('frequency(Hz)')
#axs[1].set_title('real')
axs[2].set_ylabel('Imaginary')

print(sounddata.shape)
plt.show()