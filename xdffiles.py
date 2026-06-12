# to test and see the raw eeg data being collected
import pyxdf
import matplotlib.pyplot as plt

# get file from labrecorder
streams, header = pyxdf.load_xdf("sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
eeg = streams[0]["time_series"].flatten()
timestamps = streams[0]["time_stamps"]
timestamps_ms = (timestamps - timestamps[0]) * 1000

plt.plot(timestamps_ms, eeg)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.title("Raw EEG")

# add markers at known times where I looked in different directions
plt.axvline(x=0,  color='c', linestyle='--', label='neutral')
plt.axvline(x=5000,  color='g', linestyle='--', label='look left')
plt.axvline(x=10000, color='b', linestyle='--', label='look right')
plt.axvline(x=15000, color='r', linestyle='--', label='neutral')
plt.legend()
plt.show()