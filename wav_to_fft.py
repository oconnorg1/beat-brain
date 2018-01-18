import numpy
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import rfft	#potentially faster rfft function for real signal
from scipy.io import wavfile


filename = '1000_sine.wav'	# adjust file here
print_enabled = 0	# set to 1 to print data


# read the wav
# works for 16-bit and 32-bit wav's, but not for 24-bit
print "\nReading", filename, "..."
rate, wav_data = wavfile.read(filename) # wav_data is a numpy array
print "Done reading", filename, "with sample rate", rate


# now get fft of data
# see https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
print "Computing FFT..."

#if using a two-channel track
#wav_data = wav_data.T[0]

fft_array = rfft(wav_data)	# returns an ndarray of complex numbers
# NOTE: this currently is sampling all data from the wav file (i.e. highest resolution)

print "Done computing FFT. Total samples =", len(fft_array)

# plot the fft
plt.plot(abs(fft_array))
plt.grid()
plt.show()

# print data and fft arrays (for testing)
if print_enabled:

	print "\nRaw time-domain data:"
	#numpy.set_printoptions(threshold='nan')	# to see all samples
	print wav_data, "\n"
	
	#plotting time-domain signal
	plt.plot(wav_data)
	plt.show()

	print "\n"

	print "FFT data:"
	print fft_array, "\n"


# see http://samcarcagno.altervista.org/blog/basic-sound-processing-python/
# see https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files
