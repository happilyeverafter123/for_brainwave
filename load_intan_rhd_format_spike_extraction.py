import datetime
import sys
import time

import matplotlib.pyplot as plt

import numpy as np

import os

from intanutil.header import (read_header,
                              header_to_result)
from intanutil.data import (calculate_data_size,
                            read_all_data_blocks,
                            check_end_of_file,
                            parse_data,
                            data_to_result)

from intanutil.filter import apply_notch_and_highpass_filter

from scipy.signal import find_peaks
from sklearn.decomposition import PCA

def read_data(filename):
    # Start measuring how long this read takes
    tic = time.time()

    #Open the file for reading
    with open(filename, 'rb') as fid:

        #Read header and summarize its contents o console
        header = read_header(fid)

        # Calculate how much data is present and summarize to console.
        data_present, filesize, num_blocks, num_samples = (
            calculate_data_size(header, filename, fid))

        # If .rhd file contains data, read all present data blocks into 'data'
        # dict, and verify the amount of data read.
        if data_present:
            data = read_all_data_blocks(header, num_samples, num_blocks, fid)
            check_end_of_file(filesize, fid)

    # Save information in 'header' to 'result' dict.
    result = {}
    header_to_result(header, result)

    # If .rhd file contains data, parse data into readable forms and, if
    # necessary, apply the same notch filter that was active during recording.
    if data_present:
        parse_data(header, data)
        apply_notch_and_highpass_filter(header, data)

        # Save recorded data in 'data' to 'result' dict.
        data_to_result(header, data, result)

    # Otherwise (.rhd file is just a header for One File Per Signal Type or
    # One File Per Channel data formats, in which actual data is saved in
    # separate .dat files), just return data as an empty list.
    else:
        data = []

    # Report how long read took.
    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))

    # Return 'result' dict.
    return result

def extract_analyze_and_plot_spikes(result, channel_index, threshold_ratio, pre_time, post_time, n_components):

    if 'amplifier_data' not in result or 't_amplifier' not in result:
        print("Amplifier data not found in the result dictionary.")
        return

    # Extract the signal and time
    signal = result['amplifier_data'][channel_index]
    time = result['t_amplifier']
    sample_rate = result['sample_rate']
    if sample_rate is None:
        raise ValueError("Sample rate not found in result. Ensure the header information is correctly saved.")
    
    # Convert pre_time and post_time to sample counts
    pre_samples = int((pre_time / 1000) * sample_rate)  # ms to samples.
    post_samples = int((post_time / 1000) * sample_rate)

    #convert signal so that it can be used for calculation

    # Dynamically calculate the threshold based on the maximum amplitude
    max_amplitude = np.max(signal)
    threshold = float(threshold_ratio) * max_amplitude
    print(f"Using dynamic threshold: {threshold} μV (Ratio: {threshold_ratio}, Max amplitude: {max_amplitude} μV)")

    #define the threshold range
    min_height = (1/3) * max_amplitude
    max_height = (1/2) * max_amplitude

    # Detect spikes based on the dynamic threshold range
    spikes, _ = find_peaks(signal, height=(min_height, max_height))
    print(f"Detected {len(spikes)} spikes within the range {min_height:.2f} to {max_height:.2f} μV.")

    # Extract spike waveforms
    spike_waveforms = []
    aligned_time = np.linspace(-pre_time, post_time, pre_samples + post_samples)
    for spike in spikes:
        start = max(0, spike - pre_samples)
        end = min(len(signal), spike + post_samples)
        if end - start == (pre_samples + post_samples):  # Ensure all windows are the same size
            spike_waveforms.append(signal[start:end])

    spike_waveforms = np.array(spike_waveforms)

    # Apply PCA to the spike waveforms
    pca = PCA(n_components=n_components)
    spike_features = pca.fit_transform(spike_waveforms)
    print(f"Explained variance ratio by PCA: {pca.explained_variance_ratio_}")

    return signal, time, aligned_time, spike_waveforms, spikes, spike_features, pca

if __name__ == '__main__':
    filename = sys.argv[1]
    result = read_data(filename)
    signal, time, aligned_time, spike_waveforms, spikes, spike_features, pca = extract_analyze_and_plot_spikes(result, 0, 0.3, 1, 2, 2)


    if signal is not None:
        #Save the current time
        import datetime
        currrent_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Plot spike waveforms
        plt.figure(figsize=(12, 6))
        for waveform in spike_waveforms:
            plt.plot(aligned_time, waveform, alpha=0.5, color='gray')
        plt.axvline(x=0, color='red', linestyle='--', label='Spike Time (t=0)')
        plt.title("Aligned Spike Waveforms (Channel 0)")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (μV)")
        plt.legend()
        custom_waveform_name = input("Name a graph for spike waveform (the file will be saved as: the_name_you_entered_spike_waveform_current_time.png): ")
        waveform_filename = f"{custom_waveform_name}_spike_waveform_{currrent_time}.png"
        plt.savefig(waveform_filename)
        print(f"graph is now saved: {waveform_filename}")

        # Plot original signal with spikes highlighted
        plt.figure(figsize=(60, 6))
        plt.plot(time, signal, label='Amplitude (Signal)')
        plt.scatter(time[spikes], signal[spikes], color='red', label='Detected Spikes')
        plt.title("Amplitude with Detected Spikes (Channel 0)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (μV)")
        plt.legend()
        custom_highlighted_spikes_name = input("Name a graph for highlighted spikes (the file will be saved as: the_name_you_entered_highlighted_spikes_current_time.png): ")
        highlighted_spikes_filename = f"{custom_highlighted_spikes_name}_highlighted_spikes_{currrent_time}.png"
        plt.savefig(highlighted_spikes_filename)
        print(f"graph is now saved: {highlighted_spikes_filename}")

        # Plot PCA results
        if pca.n_components == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(spike_features[:, 0], spike_features[:, 1], alpha=0.7, color='blue')
            plt.title("PCA of Spike Waveforms (2D Projection)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            custom_pca_name = input("Name a graph for pca (the file will be saved as: the_name_you_entered_pca_current_time.png): ")
            pca_filename = f"{custom_pca_name}_pca_{currrent_time}.png"
            plt.savefig(pca_filename)
            print(f"graph is now saved: {pca_filename}")
        elif pca.n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(spike_features[:, 0], spike_features[:, 1], spike_features[:, 2], alpha=0.7, color='blue')
            ax.set_title("PCA of Spike Waveforms (3D Projection)")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_zlabel("Principal Component 3")
            custom_pca_name = input("Name a graph for pca (the file will be saved as: the_name_you_entered_pca_current_time.png): ")
            pca_filename = f"{custom_pca_name}_pca_{currrent_time}.png"
            plt.savefig(pca_filename)
            print(f"graph is now saved: {pca_filename}")

    a = read_data(sys.argv[1])
    print(a)

    fig, ax = plt.subplots(2, 1)
    ax[0].set_ylabel('Amp')
    ax[0].plot(a['t_amplifier'], a['amplifier_data'][0, :])
    ax[0].margins(x=0, y=0)

    ax[1].set_ylabel('Aux')
    ax[1].plot(a['t_aux_input'], a['aux_input_data'][2, :])
    ax[1].margins(x=0, y=0)

    # ax[2].set_ylabel('Vdd')
    # ax[2].plot(a['t_supply_voltage'], a['supply_voltage_data'][0, :])
    # ax[2].margins(x=0, y=0)

    # ax[3].set_ylabel('ADC')
    # ax[3].plot(a['t_board_adc'], a['board_adc_data'][0, :])
    # ax[3].margins(x=0, y=0)

    # ax[4].set_ylabel('Digin')
    # ax[4].plot(a['t_dig'], a['board_dig_in_data'][0, :])
    # ax[4].margins(x=0, y=0)

    # ax[5].set_ylabel('Digout')
    # ax[5].plot(a['t_dig'], a['board_dig_out_data'][0, :])
    # ax[5].margins(x=0, y=0)

    # ax[6].set_ylabel('Temp')
    # ax[6].plot(a['t_temp_sensor'], a['temp_sensor_data'][0, :])
    # ax[6].margins(x=0, y=0)

    custom_name = input("Name a graph for the original brain signal graph (the file will be saved as: the_name_you_entered_original_brain_signal_graph_current_time.png): ")
    filename = f"{custom_name}_original_graph_{currrent_time}.png"
    plt.savefig(filename)
    print(f"graph is now saved: {filename}")
