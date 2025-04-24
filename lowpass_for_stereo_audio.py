#Step0: import the packages------------------------------------------------------------------------

import warnings
import numpy as np
from scipy import fftpack
from scipy import signal
from scipy.io.wavfile import read, write
from scipy.io.wavfile import WavFileWarning
from matplotlib import pyplot as plt

#ignore WavFileWarning
warnings.simplefilter("ignore", WavFileWarning)

#--------------------------------------------------------------------------------------------------

#Step1: load the data and adjust it for filtering--------------------------------------------------

#Users enter the input and output file path
input_wav_path = input("Input wav file path (e.g., 'input.wav'): ")
output_wav_path = input("Output wav file path (e.g., 'filtered_output.wav': " )

#load the wav file, and save the original data for future plotting
samplerate, data = read(input_wav_path)

#check the datatype
print(f"Original Data Type: {data.dtype}, Min: {data.min()}, Max: {data.max()}")

#if the data is int16, turn it into float 64 for filtering
if data.dtype == np.int16:
    original_dtype = 16
    print("Converting int16 data to float64 for filtering.")
    data = data.astype(np.float64)
    #scaling the amplitude from -1 to 1
    data /= np.iinfo(np.int16).max

#do the same for int32
elif data.dtype == np.int32:
    original_dtype = 32
    print("Converting int32 data to float64 for filtering.")
    data = data.astype(np.float64)
    data /= np.iinfo(np.int32).max

#if the data type is float64, do nothing
else:
    print("The data type is float64.")

#Check if the data is stereo
is_stereo = False
if data.ndim == 2 and data.shape[1] == 2:
    is_stereo = True
    print("Stereo data detected.  Applying the filter to each channel separately.")
else:
    print("Mononeural data detected.  Applying the filter directly to the data.")

#check the length of the data
print(f"Data length after conersion: {len(data)}")
#--------------------------------------------------------------------------------------------------

#Step2: Set the parameter for lowpass filtering----------------------------------------------------

x = np.arange(0, len(data)) / samplerate
fp = 1000
fs = 2000
gpass = 0.1
gstop = 30

print(f"samplerate is: {samplerate}")
#--------------------------------------------------------------------------------------------------

#Step3: Applying the lowpass filter----------------------------------------------------------------

#Butterworth Filter (Lowpass)
def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "low")
    y = signal.filtfilt(b, a, x)
    return y

#Apply the filter to each channel
if is_stereo:
    #Apply the filter to each channel separately
    filtered_left_channel = lowpass(data[:, 0], samplerate, fp, fs, gpass, gstop)
    filtered_right_channel = lowpass(data[:, 1], samplerate, fp, fs, gpass, gstop)
    #Combine them into a stereo signal
    data_filt = np.vstack((filtered_left_channel, filtered_right_channel)).T
else:
    #Apply filter to mono data
    data_filt = lowpass(data, samplerate, fp, fs, gpass, gstop)
#--------------------------------------------------------------------------------------------------

#Step4: Performing averaging Fourier transform and decibel conversion------------------------------

#overlap processing
Fs  = 4096
overlap = 90
def ov(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate
    Fc = Fs / samplerate
    x_ol = Fs * (1 - (overlap / 100))
    N_ave = int((Ts - (Fc * (overlap / 100))) / (Fc * (1 - (overlap / 100))))
    
    array = []

    #extract data
    for i in range(N_ave):
        ps = int(x_ol * i)
        array.append(data[ps:ps + Fs:1])
    return array, N_ave

#Window function processing (Hanning type)
def hanning(data_array, Fs, N_ave):
    han = signal.windows.hann(Fs)
    acf = 1 / (sum(han) / Fs)

    #process all of the overlapped waves with hanning window function
    for i in range(N_ave):
        data_array[i] = data_array[i] * han

    return data_array, acf

#FFT processing
def fft_ave(data_array, samplerate, Fs, N_ave, acf):
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2)))

    fft_axis = np.linspace(0, samplerate, Fs)
    fft_array = np.array(fft_array)
    fft_mean = np.mean(fft_array, axis = 0)

    return fft_array, fft_mean, fft_axis

#turn linear component into dB
def db(x, dBref):
    y = 20 * np.log10(x / dBref)
    return y

#Function to apply window and FFt processing for mono or stereo data
def process_channel(channel_data, samplerate, Fs, overlap):
    #Overlapping the time wave array
    t_array, N_ave = ov(channel_data, samplerate, Fs, overlap)

    #Apply window function processing
    t_array, acf = hanning(t_array, Fs, N_ave)

    #FFT processing
    fft_array, fft_mean, fft_axis = fft_ave(t_array, samplerate, Fs, N_ave, acf)

    #convert to dB scale
    fft_mean = db(fft_mean, 2e-5)
    return fft_array, fft_mean, fft_axis

#process FFT for stereo or mono data
if is_stereo:
    #process each channel separately
    fft_array_org_left, fft_mean_org_left, fft_axis_org_left = process_channel(data[:, 0], samplerate, Fs, overlap)
    fft_array_org_right, fft_mean_org_right, fft_axis_org_right = process_channel(data[:, 1], samplerate, Fs, overlap)

    #Average the two channels for plotting
    fft_array_org = (fft_array_org_left + fft_array_org_right) / 2
    fft_mean_org = (fft_mean_org_left + fft_mean_org_right) / 2
    fft_axis_org = (fft_axis_org_left + fft_axis_org_right) / 2

else:
    fft_array_org, fft_mean_org, fft_axis_org = process_channel(data, samplerate, Fs, overlap)

#process FFT for filtered data
if is_stereo:
    #process each channel separately
    fft_array_filt_left, fft_mean_filt_left, fft_axis_filt_left = process_channel(data_filt[:, 0], samplerate, Fs, overlap)
    fft_array_filt_right, fft_mean_filt_right, fft_axis_filt_right = process_channel(data_filt[:, 1], samplerate, Fs, overlap)

    #Average the two channels for plotting
    fft_array_filt = (fft_array_filt_left + fft_array_filt_right) / 2
    fft_mean_filt = (fft_mean_filt_left + fft_mean_filt_right) / 2
    fft_axis_filt = (fft_axis_filt_left + fft_axis_filt_right) / 2
else:
    fft_array_filt, fft_mean_filt, fft_axis_filt = process_channel(data_filt, samplerate, Fs, overlap)
#--------------------------------------------------------------------------------------------------

#Step5: adjusting the scaling----------------------------------------------------------------------

data_filt /= np.max(np.abs(data_filt))

if original_dtype == 16:
    data_filt *= np.iinfo(np.int16).max
    #turn the datatype back to int16
    data_filt = data_filt.astype(np.int16)

elif original_dtype == 32:
    data_filt *= np.iinfo(np.int32).max
    #turn the datatype back to int32
    data_filt = data_filt.astype(np.int32)
#--------------------------------------------------------------------------------------------------

#Step6: save the filtered data---------------------------------------------------------------------

write(output_wav_path, samplerate, data_filt)

print(f"Filtered audio saved to {output_wav_path}")
#--------------------------------------------------------------------------------------------------
