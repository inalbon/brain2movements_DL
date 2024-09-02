import matplotlib.pyplot as plt
import numpy as np

import elephant.statistics as es
import elephant.kernels as ek
import neo
import quantities as pq


def calculate_spiketrain_of_interest(units_df, neuron_id, window):
    spiketimes = units_df['spike_times'][neuron_id]
    spiketimes_of_interest = spiketimes[(spiketimes >= window[0]) & (spiketimes <= window[1])]
    spiketrain = neo.SpikeTrain(spiketimes_of_interest*pq.s, t_stop=window[1])

    return spiketrain


def calculate_firing_rate_of_interest(units_df, neuron_id, window, kernel=ek.GaussianKernel(sigma=50*pq.ms)):
    bin_size = 5*pq.ms

    spiketrain = calculate_spiketrain_of_interest(units_df, neuron_id, window)
    firing_rate = es.instantaneous_rate(spiketrain, sampling_period=bin_size, t_start=window[0], t_stop=window[1], kernel=kernel)

    return firing_rate


def plot_spike_trains(spiketrain, window):
    plt.figure(figsize=(8, 2))
    plt.eventplot(spiketrain.magnitude, colors='black', linelengths=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Spikes')
    plt.title('Spike Train')
    plt.xlim([window[0], window[1]])
    plt.grid(True)
    plt.tight_layout()

    return None


def plot_firing_rate(firing_rate, window, label=None):
    plt.figure(figsize=(8, 2))
    plt.plot(firing_rate.times, firing_rate.magnitude, label=label, color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Rate (Hz)')
    plt.title('Firing Rate')
    plt.xlim([window[0], window[1]])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return None

def plot_binned_firing_rate(binned_firing_rate, window, bin_size, label=None):
    plt.figure(figsize=(8, 2))
    plt.xlabel('Time (s)')
    plt.ylabel('Rate (Hz)')
    plt.title('Firing Rate')
    plt.bar(binned_firing_rate.times.rescale('s'), binned_firing_rate.magnitude.flatten(), align='center', label=label, color='blue', width=bin_size.rescale('s').magnitude)
    plt.xlim([window[0], window[1]])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return None


def plot_spike_raster(spiketrain_list, window):
    for i, spiketrain in enumerate(spiketrain_list):
        t = spiketrain.rescale(pq.s)
        plt.plot(t, i * np.ones_like(t), 'k.', markersize=2)

    plt.axis('tight')
    plt.xlim(window[0], window[1])
    plt.xlabel('Time (s)')
    plt.ylabel('Spike Train Index')
    plt.title('Spike Raster')
    plt.gca().tick_params(axis='both', which='major')
    plt.show()

    return None
