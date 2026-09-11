"""
BCI-Music Eye Movement Threshold Detector
--------------------------------------
The detector responds to the CHANGE into an eye movement,
rather than a gaze that is already being held.

"""

from collections import deque
import time

import numpy as np
import matplotlib.pyplot as plt
from pylsl import (
    StreamInlet,
    StreamInfo,
    StreamOutlet,
    resolve_streams,
)
from scipy.signal import butter, lfilter, lfilter_zi


# ============================================================
# STREAM SETTINGS
# ============================================================

EEG_STREAM_TYPE = "EEG"

CONTROL_STREAM_NAME = "IDUN_Stream"
CONTROL_STREAM_TYPE = "Markers"
CONTROL_SOURCE_ID = "bci_music_eye_controls"


# ============================================================
# SIGNAL SETTINGS
# ============================================================

SAMPLE_RATE = 250

# Live detector window.
ONSET_WINDOW_SAMPLES = 125          # 0.5 seconds

# Calibration timings.
BASELINE_SECONDS = 10.0              # straight-ahead baseline
MOVEMENT_RECORDING_SECONDS = 10.0    # gives user time to react

# How many LEFT / RIGHT trials to collect and average during calibration.
CALIBRATION_TRIALS_PER_DIRECTION = 3 #can change

# Original detector threshold settings.
THRESHOLD_SCALE = 0.65

# 1 second cooldown at 250 Hz.
COOLDOWN_SAMPLES = 250

NEUTRAL_WINDOW_SAMPLES = 500  # 2 seconds of neutral signal for baseline noise
PLOT_CALIBRATION_DEBUG = True  # show calibration diagnostic plots, set to false for faster 

# ============================================================
# GAZE STATE MACHINE SETTINGS
# ============================================================

GAZE_NEUTRAL = "NEUTRAL"
GAZE_LEFT = "LEFT"
GAZE_RIGHT = "RIGHT"

# How long to ignore detection after a LEFT or RIGHT command.
# This gives the user time to return their eyes to center.
GAZE_HOLD_SECONDS = 10.0  #can change depending on user

GAZE_HOLD_SAMPLES = int(
    GAZE_HOLD_SECONDS * SAMPLE_RATE
)

#MIN_AWAY_SAMPLES = int(0.50 * SAMPLE_RATE)
RETURN_SETTLE_SAMPLES = int(0.40 * SAMPLE_RATE)

# Calibration locks onto the FIRST meaningful departure from neutral
# rather than the largest event anywhere in the full movement recording
# so user just needs to lookin the correct direction once, can hold or not
CALIBRATION_SEARCH_SECONDS = 3.0 # searches for movement in first 3 seconds
CALIBRATION_LOCK_SECONDS = 0.40 # once movement detected, next .4 seconds used to find that peak so user doesn't have to sustain
CALIBRATION_DEPARTURE_MAD_MULTIPLIER = 4.0 # this checks if the movement is significant enough to be considered a departure from neutral
#if not then the largest movement in the first 3 seconds is used instead


# ============================================================
# FIND EEG STREAM
# ============================================================

def find_eeg_inlet(wait_time=5.0):
    """
    Look for an already-running LSL stream with type='EEG'
    and create an inlet for receiving samples.
    """

    print("Looking for EEG stream...", flush=True)

    streams = resolve_streams(wait_time=wait_time)

    eeg_streams = [
        s for s in streams
        if s.type() == EEG_STREAM_TYPE
    ]

    if not eeg_streams:

        available = ", ".join(
            f"{s.name()} [{s.type()}]"
            for s in streams
        ) or "none"

        raise RuntimeError(
            "No LSL stream with type='EEG' was found. "
            "Start the IDUN streaming program first. "
            f"Available streams: {available}"
        )

    stream = eeg_streams[0]

    print(
        f"EEG stream found: {stream.name()} "
        f"({stream.channel_count()} channel(s), "
        f"{stream.nominal_srate():.0f} Hz)\n",
        flush=True,
    )

    return StreamInlet(stream)


# ============================================================
# CREATE OUTPUT CONTROL STREAM
# ============================================================

def create_control_outlet():
    """
    Create an LSL marker stream that sends:

        LEFT
        RIGHT

    The game/application (frontend.py) can listen to this stream.
    """

    info = StreamInfo(
        CONTROL_STREAM_NAME,
        CONTROL_STREAM_TYPE,
        1,
        0,
        "string",
        CONTROL_SOURCE_ID,
    )

    outlet = StreamOutlet(info)

    print(
        f"BCI control stream started: "
        f"{CONTROL_STREAM_NAME}\n",
        flush=True,
    )

    return outlet


# ============================================================
# HIGH-PASS FILTER
# ============================================================

def make_highpass_filter(  #to be used on samples 
    cutoff=0.5,
    fs=SAMPLE_RATE,
    order=2,
):
    """
    Create a stateful high-pass filter.

    The filter state is preserved between samples so filtering
    behaves continuously.
    """

    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype="highpass",
    )

    zi = lfilter_zi(b, a)

    return b, a, zi #this creates filter values


# ============================================================
# GENERAL HELPERS
# ============================================================

def countdown(seconds=3):
    """
    Print a countdown to use before calibration.
    """

    for i in range(seconds, 0, -1):
        print(f"{i}...", flush=True)
        time.sleep(1)


def drain_buffer(inlet):
    """
    Remove samples currently waiting in the LSL inlet.

    This keeps old samples from accidentally becoming part
    of the next trial. - the buffer is drained so movement is continuous
    """

    while True:

        sample, _ = inlet.pull_sample(
            timeout=0.0
        )

        if sample is None:
            break


# ============================================================
# FILTERED SAMPLE COLLECTION
# ============================================================

def collect_filtered_samples(     # collected both raw and filtered samples
    inlet,
    num_samples,
    hp_filter,
):
    """
    Collect a fixed number of EEG samples.

    Returns:
        raw_samples
        filtered_samples
        updated hp_filter
    """

    raw_samples = []          # empty list
    filtered_samples = []      # empty list

    b, a, zi = hp_filter     # call filter

    while len(filtered_samples) < num_samples: # filtered sample length is less than the actual samples

        sample, _ = inlet.pull_sample(
            timeout=1.0
        )

        if sample is None:
            continue

        raw_value = sample[0]  # raw value is straight from the collected inlet samples

        filtered, zi = lfilter(
            b,
            a,
            [raw_value],
            zi=zi,
        )

        raw_samples.append(raw_value)
        filtered_samples.append(filtered[0])

    return (
        np.asarray(raw_samples),    # return samples as array
        np.asarray(filtered_samples),
        (b, a, zi),
    )


# ============================================================
# BASIC ONSET MEASUREMENT
# ============================================================

def onset_measurement(window):   # where the signal is in the collection window
    """
    Measure the strongest signed deflection in a window.

    The first portion of the window acts as a local reference since user doens't look immediately

    Returns:
        signed_peak
        positive_peak
        negative_peak 
    """

    window = np.asarray(  
        window, 
        dtype=float,
    )

    if len(window) < 5:  #less than 5 seconds because not enough time for baseline and deflection
        raise ValueError(
            "Onset window is too short."      
        )

    # Use first 20% as the local reference.- why? - so that only the beginning is the baseline
    reference_samples = max(
        5,
        int(0.20 * len(window)),
    )

    reference = np.median(
        window[:reference_samples] # take the median sample as the baseline (could be the average?-would this be better)
    )

    relative = (
        window - reference  # relative is the difference between the window and the reference (baseline)  
    )

    positive_peak = np.max(relative) # this is max deflection from the baseline
    negative_peak = np.min(relative)  # min deflection from the baseline 

    # Keep the sign of whichever deflection is larger.
    if abs(positive_peak) >= abs(negative_peak):
        signed_peak = positive_peak
    else:
        signed_peak = negative_peak

    return (
        signed_peak,
        positive_peak,
        negative_peak,
    )


# ============================================================
# STREAMING (used by calibration and live detection)
# ============================================================

def streaming_onset_step(
    value,  # newest sample coming in from headset
    onset_buffer, # baseline corrected signal (rolling)
    neutral_buffer, # neutral/straight ahead samplees (rolling) to use as basline or zero
):
    """
    Advance the online detector by exactly one sample. Does not decide left vs right, 
    but prepares and measures signal.
    Returns:
        signed_peak, or None if onset_buffer isn't full yet.
    """

    neutral_baseline = (   #calculate neutral baseline
        np.median(neutral_buffer)   #take median of neutral baseline samples from neutral buffer
        if len(neutral_buffer) > 0
        else value  #if neutral buffer empty, use current sample for baseline
    )

    corrected_value = ( # subtract neutral baseline from value to find how many units above neutral signal is
        value
        - neutral_baseline
    )   #this moves neutral to zero no matter the value

    onset_buffer.append(  #corrected samples is added to onset buffer
        corrected_value   #value appened to window sliding along live signal
    )

    if len(onset_buffer) < onset_buffer.maxlen: #checking whether enough samples collected before analyzing
        return None

    signed_peak, _, _ = onset_measurement( #analyze compelted onset window once buffer is fill
        onset_buffer  # this examines the window and determines onset deflection
    )

    return signed_peak  #send first result, give measurement back to dectector


# analysis step occuring inside calibration t ofind threshold and how large peaks have to be
#from threshold to count
def replay_trial_through_streaming_detector( #handles entire calibration trial
    baseline_filtered_samples,  #where user is looking straight ahead
    movement_filtered_samples, # where user looked after cue in calibration (L or R)
):
    """
    Replay one calibration trial through the same streaming onset
    calculation used during live detection.

    Calibration locks onto the FIRST meaningful departure from NEUTRAL.
    Once that departure is found, its direction is locked and only a short
    same-direction peak window is considered. Later opposite return snaps
    are ignored.
    """

    #create rolling buffers
    onset_buffer = deque(maxlen=ONSET_WINDOW_SAMPLES)  # deque is a list that holds recent correctd samples to calculate onset
    neutral_buffer = deque(maxlen=NEUTRAL_WINDOW_SAMPLES)  #hold recent neutral samples to estimate baseline

    # Replay neutral baseline and collect the normal streaming-onset statistic.
    baseline_signed_peaks = [] #store onset measurements from neutral gaze 
    #to make sure normal movements aren't seen as eye movements

    for value in baseline_filtered_samples: #go through neutral recording one sample at a time, value becomes each sample one after another
        signed_peak = streaming_onset_step( #taking pre recorded calibration signal as live
            value,
            onset_buffer,
            neutral_buffer,
        )

        if signed_peak is not None:
            baseline_signed_peaks.append(float(signed_peak)) #once valid onset measurement exists save it

        neutral_buffer.append(value) # raw filtered neutral sample into neutral buffer show detector what current straight-ahead baseline looks like

    # Robust departure threshold from neutral variation- how big a movement
    # has to be to count
    if baseline_signed_peaks: #if successfully collected neutral onset measurements, calculate threshold
        baseline_abs = np.abs(
            np.asarray(baseline_signed_peaks, dtype=float) #remove signs to read how large normal noise is in neutral
        )

        #find median size of neutral fluctuations
        baseline_center = float(np.median(baseline_abs))
        baseline_mad = float(  #median absolute deviation - like standard deviation
            np.median(np.abs(baseline_abs - baseline_center)) #how spread out are the normal neutral values around typical neutral value
        )
        robust_sigma = 1.4826 * baseline_mad #convert median absolute deviation to standard deviation equivalent
        #good so outliers don't mess up baseline threshold

        departure_threshold = ( #threshold = normal neutral level + safety margin
            baseline_center
            + CALIBRATION_DEPARTURE_MAD_MULTIPLIER * robust_sigma
        )
    else:
        departure_threshold = 0.0 #if no baseline peaks

    departure_threshold = max(float(departure_threshold), 1e-6) #prevents threshold from actually being zero (always slightly positive)

    movement_signed_peaks = [] #preparing to search for eye movements

    # start as empty since no movements yet
    departure_index = None #where first meaningful movement occurs in the recording
    locked_direction = None # what sign first movement is
    locked_peak = None # strongest peak in that direction
    peak_index = None # where strongest peak happened
    lock_end_index = None # how long to look for strongest peak

    search_samples = min(  # set how long to search for initial eye movements after cue
        len(movement_filtered_samples),
        int(CALIBRATION_SEARCH_SECONDS * SAMPLE_RATE), #750 samples
    )

    #how many samples examined after first departure
    lock_samples = max(
        1,
        int(CALIBRATION_LOCK_SECONDS * SAMPLE_RATE), #only look for 100 samples after first departure
    )

    # Replay movement with neutral baseline frozen- does not update baseline with each movement so no new neutral
    # than original one found from calibration
    for index, value in enumerate(movement_filtered_samples):
        signed_peak = streaming_onset_step(
            value,
            onset_buffer,
            neutral_buffer,
        )

        #guarantees signed_valus is always a number
        signed_value = 0.0 if signed_peak is None else float(signed_peak)
        movement_signed_peaks.append(signed_value)

        # Before departure: only search the early post-cue period.
        if departure_index is None:
            if index >= search_samples:
                break #If you've reached the end of your allowed search period without finding anything, stop searching.

            #checking if valid onsent measurement and if larger than neutral noise
            if (
                signed_peak is not None
                and abs(signed_value) >= departure_threshold
            ):
                departure_index = index #save first meaningful departure
                locked_direction = ( #detemine direction signal went (signal NOT eye direction)
                    "positive" if signed_value >= 0 else "negative"
                )
                locked_peak = signed_value #first signal is best peak found
                peak_index = index
                lock_end_index = min( #creat ime windw after initial departure
                    len(movement_filtered_samples) - 1, #can look for stronger same-direction peaks
                    index + lock_samples,
                )

            continue #Move to next sample

        # After departure: only strengthen the same-direction onset.
        # prevents going back to neutral as being a direction change
        same_direction = ( #is new value in same direction as OG movement
            (locked_direction == "positive" and signed_value > 0)
            or
            (locked_direction == "negative" and signed_value < 0)
        )

        if ( #if still pointing in same and stronger than previous peak, update peak
            same_direction
            and abs(signed_value) > abs(locked_peak)
        ):
            locked_peak = signed_value
            peak_index = index  #save new peak value

        if index >= lock_end_index: #once short lock window finished, stop so calibration focuses on first eye movement
            break

    movement_signed_peaks = np.asarray(
        movement_signed_peaks,
        dtype=float,
    )

    # Fallback: if no clear departure crossed the robust threshold,
    # use the largest event only within the early search window.
    if departure_index is None:
        fallback_trace = movement_signed_peaks[:search_samples]

        if len(fallback_trace) == 0:
            raise RuntimeError(
                "Calibration movement recording was empty."
            )

        peak_index = int( #finds largest absolute signal in early window
            np.argmax(np.abs(fallback_trace))
        )
        locked_peak = float(fallback_trace[peak_index])
        departure_index = peak_index #use this event as calibration peak

        print(
            "WARNING: no clear first departure crossed the calibration "
            "threshold; using the largest event in the early search window.",
            flush=True,
        )

    return (
        float(locked_peak), #How storn detected eye movement was and if +/-
        int(peak_index), # what sample the strongest onset/peak happened
        movement_signed_peaks, # whole detector output trace during movement trial
        float(departure_threshold), # how large a movement needed to be before counting as movement and not just noise
        int(departure_index), #what sample signal first crossed thresholds
    )


# ============================================================
# COLLECT ONE LEFT OR RIGHT CALIBRATION TRIAL
# ============================================================

def collect_single_movement_trial(
    inlet, #where samples come from
    direction_name, # L or R
    trial_number, # which trial we are on
    total_trials, 
    hp_filter, 
):
    """
    Collect ONE LEFT or RIGHT movement trial.

    The user first looks straight ahead.

    Then:
        LOOK LEFT / LOOK RIGHT

    A long recording is collected so the user has enough time
    to react naturally.

    The recording is then replayed through the exact same
    streaming detector used during live use (replay_trial_through_streaming_detector), so the peak this
    trial reports is the peak live detection would actually see.
    """

    print(
        "\n----------------------------------",
        flush=True,
    )

    print(
        f"{direction_name} CALIBRATION "
        f"(trial {trial_number}/{total_trials})",
        flush=True,
    )

    print(
        "----------------------------------",
        flush=True,
    )

    # ========================================================
    # 1. USER STARTS FROM STRAIGHT AHEAD
    # ========================================================

    print(
        "\nFirst, look STRAIGHT AHEAD.",
        flush=True,
    )

    countdown(3)

    drain_buffer(inlet) #throws away samples already in LSL buffer

    print(
        "Measuring straight-ahead baseline...",
        flush=True,
    )

    # decide how many baseline samples to record
    #converts samples into seconds
    baseline_num_samples = int(
        BASELINE_SECONDS
        * SAMPLE_RATE
    )

    #collect samples- raw eeg values, hp filtered eeg values, and updated filter state
    baseline_raw_samples, baseline_samples, hp_filter = (
        collect_filtered_samples(
            inlet,
            baseline_num_samples,
            hp_filter,
        )
    )

    # Median protects against occasional noisy samples.
    #find typical baseline 
    baseline_value = float(
        np.median(
            baseline_samples
        )
    )

    #how much signal moved while user was in neutral, good for calibration quality
    baseline_noise = float(
        np.std(
            baseline_samples
        )
    )

    print(
        f"Local baseline: {baseline_value:.2f}",
        flush=True,
    )

    # ========================================================
    # 2. GIVE MOVEMENT CUE
    # ========================================================

    print(
        f"\nLOOK {direction_name} NOW!",
        flush=True,
    )

    # ========================================================
    # 3. RECORD LONG ENOUGH FOR NATURAL REACTION
    # ========================================================

    movement_num_samples = int(
        MOVEMENT_RECORDING_SECONDS
        * SAMPLE_RATE
    )

    #record eye movement samples- first part of signal represents eye movement
    movement_raw_samples, movement_samples, hp_filter = (
        collect_filtered_samples(
            inlet,
            movement_num_samples,
            hp_filter,
        )
    )

    # ========================================================
    # 4. REPLAY THIS TRIAL THROUGH THE SAME STREAMING DETECTOR
    #    THAT LIVE DETECTION USES 
    # ========================================================
    #the peak measured here is the peak live detection will actually compute for
    # this movement, not an idealized offline estimate
    (
        signed_peak,
        peak_index,
        movement_signed_peaks,
        departure_threshold,
        departure_index,
    ) = replay_trial_through_streaming_detector( #to allow for determination of actual eye movement
        baseline_filtered_samples=baseline_samples,
        movement_filtered_samples=movement_samples,
    )

    #convert indexes into time
    peak_time = (
        peak_index
        / SAMPLE_RATE
    )

    #where first meaningful movement began
    departure_time = (
        departure_index
        / SAMPLE_RATE
    )

    print(
        f"\n{direction_name} trial {trial_number} "
        f"first departure: {departure_time:.3f} s after cue",
        flush=True,
    )

    print(
        f"  locked streaming peak = {signed_peak:.2f} "
        f"at {peak_time:.3f} s",
        flush=True,
    )

    print(
        f"  departure threshold = {departure_threshold:.2f}",
        flush=True,
    )

    # print(
    #     f"  baseline noise SD = "
    #     f"{baseline_noise:.2f}\n",
    #     flush=True,
    # )

    return {
        "signed_peak": signed_peak,
        "baseline_noise": baseline_noise,
        "hp_filter": hp_filter,
        "baseline_raw": baseline_raw_samples,
        "baseline_filtered": baseline_samples,
        "raw": movement_raw_samples,
        "filtered": movement_samples,
        "streaming_peaks": movement_signed_peaks,
        "peak_index": peak_index,
        "departure_index": departure_index,
        "departure_threshold": departure_threshold,
    }


# ============================================================
# COLLECT MULTIPLE TRIALS FOR ONE DIRECTION AND AGGREGATE
# ============================================================

def collect_movement_calibration(
    inlet,
    direction_name,
    hp_filter,
    num_trials=CALIBRATION_TRIALS_PER_DIRECTION,
):
    """
    Collect several LEFT or RIGHT trials and aggregate them.
    i.e All calibration trials for one direction

    A single trial's amplitude is noisy -- one unusually big or
    small signal would set the threshold for the whole
    session.
     
    Taking the median signed peak and noise across several
    trials makes the calibration much less sensitive to any one
    trial.

    Returns:
        aggregated_signed_peak (median across trials)
        aggregated_baseline_noise (median across trials)
        updated hp_filter
        list of per-trial result dicts (for debug plotting)
    """

    trials = [] #store all trials from same direction

    for trial_number in range(1, num_trials + 1): #repeat single-trial function

        trial = collect_single_movement_trial( #calls function to go through individual trials of each direction
            inlet=inlet,
            direction_name=direction_name,
            trial_number=trial_number,
            total_trials=num_trials,
            hp_filter=hp_filter,
        )

        hp_filter = trial["hp_filter"] #use new filter state for next trial (save)

        trials.append(trial)  # add trial results to list

        if trial_number < num_trials: # between trials return eyes to center

            print(
                "Return eyes to center.",
                flush=True,
            )

            time.sleep(1.5)

            drain_buffer(inlet)

    #pull out each trial's peak
    signed_peaks = [
        trial["signed_peak"]
        for trial in trials
    ]

    #pull out baseline noise values
    baseline_noises = [
        trial["baseline_noise"]
        for trial in trials
    ]

    #take median across trials to get typical calibration value for live detection reference
    aggregated_signed_peak = float(
        np.median(signed_peaks)
    )

    #same thing but for baseline noises to know differences between noise and actual movement
    aggregated_baseline_noise = float(
        np.median(baseline_noises)
    )

    print(
        f"{direction_name} aggregated over {num_trials} trials:",
        flush=True,
    )

    print(
        f"  per-trial signed peaks = "
        f"{[round(p, 2) for p in signed_peaks]}",
        flush=True,
    )

    print(
        f"  median signed peak = "
        f"{aggregated_signed_peak:.2f}\n",
        flush=True,
    )

    return (
        aggregated_signed_peak,
        aggregated_baseline_noise,
        hp_filter,
        trials,
    )


def plot_calibration_debug(
    left_trials,
    right_trials,
):
    """
    Overlay each LEFT/RIGHT trial's raw and filtered signal,
    plus the streaming signed-peak trace that
    replay_trial_through_streaming_detector produced for it --
    i.e. the actual statistic live detection computes -- so
    calibration quality and trial-to-trial consistency can be
    checked visually.

    A vertical line marks the largest streaming peak per trial
    (this is what feeds the median used to set the threshold).
    """

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 10),
        sharex="col",
    )

    def plot_direction(trials, column, label):

        for trial_index, trial in enumerate(trials):

            raw_time_axis = (
                np.arange(len(trial["raw"]))
                / SAMPLE_RATE
            )

            streaming_time_axis = (
                np.arange(len(trial["streaming_peaks"]))
                / SAMPLE_RATE
            )

            trial_label = f"trial {trial_index + 1}"

            axes[0, column].plot(
                raw_time_axis,
                trial["raw"],
                label=trial_label,
            )

            axes[1, column].plot(
                raw_time_axis,
                trial["filtered"],
                label=trial_label,
            )

            axes[2, column].plot(
                streaming_time_axis,
                trial["streaming_peaks"],
                label=trial_label,
            )

            axes[2, column].axvline(
                trial["departure_index"] / SAMPLE_RATE,
                linestyle="--",
                alpha=0.6,
            )

            axes[2, column].axvline(
                trial["peak_index"] / SAMPLE_RATE,
                linestyle=":",
                alpha=0.6,
            )

        axes[0, column].set_title(f"{label} - Raw")
        axes[1, column].set_title(f"{label} - High-pass filtered")
        axes[2, column].set_title(f"{label} - Streaming signed peak")
        axes[2, column].axhline(0, linestyle="--", color="black")
        axes[2, column].set_xlabel("Time (s)")
        axes[2, column].legend(fontsize=8)

    plot_direction(left_trials, 0, "LEFT")
    plot_direction(right_trials, 1, "RIGHT")

    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel("Amplitude")
    axes[2, 0].set_ylabel("Signed peak")

    plt.tight_layout()
    plt.show()


# ============================================================
# FULL CALIBRATION
# ============================================================

def calibrate(inlet):
    """
    Calibration order:

        1. STRAIGHT -> LEFT, repeated CALIBRATION_TRIALS_PER_DIRECTION times.
        2. STRAIGHT -> RIGHT, repeated CALIBRATION_TRIALS_PER_DIRECTION times.

    LEFT and RIGHT are measured relative to the immediately
    preceding straight-ahead baseline, and each direction's final
    threshold is based on the MEDIAN of several trials rather than
    a single trial.

    The user does NOT have to react instantly.
    """

    print(
        "\n==================================",
        flush=True,
    )

    print(
        "MUSIC SELECTION EYE MOVEMENT CALIBRATION",
        flush=True,
    )

    print(
        "==================================\n",
        flush=True,
    )

    print(
        "For LEFT and RIGHT trials:",
        flush=True,
    )

    print(
        "1. Start by looking straight ahead.",
        flush=True,
    )

    print(
        "2. Wait for the movement cue.",
        flush=True,
    )

    print(
        "3. Move your eyes naturally when you see it.",
        flush=True,
    )

    print(
        f"You'll do {CALIBRATION_TRIALS_PER_DIRECTION} trials per "
        f"direction. You do NOT need to react instantly.\n",
        flush=True,
    )

    hp_filter = make_highpass_filter()

    print(
        "Warming up filter...",
        flush=True,
    )

    _, _, hp_filter = collect_filtered_samples(
        inlet,
        SAMPLE_RATE * 2,   # 2 seconds
        hp_filter,
    )

    print(
        "Filter settled.\n",
        flush=True,
    )

    # ========================================================
    # PART 1: LEFT (multiple trials)
    # ========================================================

    (
        left_peak,
        _left_baseline_noise,
        hp_filter,
        left_trials,
    ) = collect_movement_calibration( # record neutral, movement and then replay through detector
        inlet=inlet,
        direction_name="LEFT",
        hp_filter=hp_filter,
    )

    print(
        "Return eyes to center.",
        flush=True,
    )

    time.sleep(1.5)

    drain_buffer(inlet)

    # ========================================================
    # PART 2: RIGHT (multiple trials)
    # ========================================================

    (
        right_peak,
        _right_baseline_noise,
        hp_filter,
        right_trials,
    ) = collect_movement_calibration(
        inlet=inlet,
        direction_name="RIGHT",
        hp_filter=hp_filter,
    )

    if PLOT_CALIBRATION_DEBUG:

        plot_calibration_debug(
            left_trials=left_trials,
            right_trials=right_trials,
        )

    # ========================================================
    # BUILD DETECTION THRESHOLDS
    # ========================================================

    #determine sign for left
    left_direction = (
        "positive"
        if left_peak >= 0
        else "negative"
    )

    #determine sign for right
    right_direction = (
        "positive"
        if right_peak >= 0
        else "negative"
    )

    #build live thresholds- only 65% of max peak calculated needed to activate
    left_threshold = float(
        abs(left_peak) * THRESHOLD_SCALE
    )

    right_threshold = float(
        abs(right_peak) * THRESHOLD_SCALE
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n==================================",
        flush=True,
    )

    print(
        "CALIBRATION COMPLETE",
        flush=True,
    )

    print(
        "==================================",
        flush=True,
    )

    print(
        f"LEFT:",
        flush=True,
    )

    print(
        f"  direction = {left_direction}",
        flush=True,
    )

    print(
        f"  median peak = {left_peak:.2f}",
        flush=True,
    )

    print(
        f"  threshold = "
        f"{left_threshold:.2f}",
        flush=True,
    )

    print(
        f"\nRIGHT:",
        flush=True,
    )

    print(
        f"  direction = {right_direction}",
        flush=True,
    )

    print(
        f"  median peak = {right_peak:.2f}",
        flush=True,
    )

    print(
        f"  threshold = "
        f"{right_threshold:.2f}",
        flush=True,
    )

    # ========================================================
    # CHECK FOR CALIBRATION SIGN ERRORS
    # ========================================================

    if left_direction == right_direction:

        print(
            "\nWARNING:"
            "\nLEFT and RIGHT produced the SAME "
            "deflection direction." \
            "\nConsider re-calibrating",
            flush=True,
        )

    else:

        print(
            "\nGood: LEFT and RIGHT produced "
            "opposite deflection directions.",
            flush=True,
        )

    print()

    # Return the filter too so live detection can continue
    # with the same filter state instead of resetting it.
    calibration = {
        "left_direction": left_direction,
        "right_direction": right_direction,
        "left_threshold": left_threshold,
        "right_threshold": right_threshold,
    }

    return (
        calibration,
        hp_filter,
    )


# ============================================================
# THRESHOLD CHECK FOR LIVE DETECTION
# ============================================================

def threshold_crossed(
    signed_peak,
    direction,
    threshold,
):
    """
    Determine whether a signal crossed the calibrated threshold
    in the expected direction.
    """

    if direction == "positive":

        return (
            signed_peak
            >= threshold
        )

    return (
        signed_peak
        <= -threshold
    )


# ============================================================
# LIVE DETECTION
# ============================================================

def run_detector(
    inlet,
    control_outlet,
    calibration,
    hp_filter,
):
    """
    Detect gaze transitions with a three-state state machine.

    Commands fire only when leaving NEUTRAL:
        NEUTRAL -> LEFT   sends LEFT
        NEUTRAL -> RIGHT  sends RIGHT
        Back to Neutral -> nothing sent or measured here

    """

    # recreate two rolling buffers    
    onset_buffer = deque(maxlen=ONSET_WINDOW_SAMPLES) #what is straight ahead currently
    neutral_buffer = deque(maxlen=NEUTRAL_WINDOW_SAMPLES) #what is recent signal doing

    # start in neutral, assume eyes in center
    gaze_state = GAZE_NEUTRAL
    samples_in_away_state = 0
    neutral_settle_counter = 0

    print(
        "Collecting rolling neutral baseline...",
        flush=True,
    )

    # --------------------------------------------------------
    # Initial rolling neutral baseline
    # --------------------------------------------------------

    #collect fresh live neutral baseline - fill buffer before detection begins
    while len(neutral_buffer) < NEUTRAL_WINDOW_SAMPLES:
        sample, _ = inlet.pull_sample(timeout=1.0) #pull sample
  
        if sample is None:
            continue

        b, a, zi = hp_filter    # unpack filter
        filtered, zi = lfilter( # filter one EEG sample
            b,
            a,
            [sample[0]],
            zi=zi,
        )
        hp_filter = (b, a, zi) # store updated filter state

        neutral_buffer.append(filtered[0])

    print(
        "Rolling neutral baseline collected.",
        flush=True,
    )

    # --------------------------------------------------------
    # Fill initial onset buffer while neutral.
    # --------------------------------------------------------
    print(
        "Collecting initial live window...",
        flush=True,
    )

    while len(onset_buffer) < ONSET_WINDOW_SAMPLES:
        sample, _ = inlet.pull_sample(timeout=1.0)

        if sample is None:
            continue

        b, a, zi = hp_filter
        filtered, zi = lfilter(
            b,
            a,
            [sample[0]],
            zi=zi,
        )
        hp_filter = (b, a, zi)

        value = filtered[0]

        streaming_onset_step(
            value,
            onset_buffer,
            neutral_buffer,
        )

        neutral_buffer.append(value)

    print(
        "Starting onset threshold detection...",
        flush=True,
    )
    print(
        "Initial gaze state: NEUTRAL\n",
        flush=True,
    )

    # --------------------------------------------------------
    # Live gaze-state machine.
    # --------------------------------------------------------
    while True:
        sample, _ = inlet.pull_sample(timeout=1.0)

        if sample is None:
            continue

        b, a, zi = hp_filter
        filtered, zi = lfilter(
            b,
            a,
            [sample[0]],
            zi=zi,
        )
        hp_filter = (b, a, zi)

        value = filtered[0]

        signed_peak = streaming_onset_step( #same as during calibration
            value,
            onset_buffer,
            neutral_buffer,
        )

        if signed_peak is None:
            continue

        # does this like calibrated LEFT
        left_crossed = threshold_crossed(
            signed_peak,
            calibration["left_direction"],
            calibration["left_threshold"],
        )

        # is it calibrated RIGHT
        right_crossed = threshold_crossed(
            signed_peak,
            calibration["right_direction"],
            calibration["right_threshold"],
        )

        # Defensive tie-break if both somehow cross - likely not used 
        if left_crossed and right_crossed:
            left_score = (
                abs(signed_peak)
                / calibration["left_threshold"]
            )
            right_score = (
                abs(signed_peak)
                / calibration["right_threshold"]
            )

            if left_score >= right_score:
                right_crossed = False
            else:
                left_crossed = False

        # ====================================================
        # NEUTRAL: only state that can emit a command.
        # ====================================================
        if gaze_state == GAZE_NEUTRAL: # only way detector will let another command happen
            if neutral_settle_counter > 0:
                neutral_settle_counter -= 1
                continue

            if left_crossed: #if live signal matches left pattern from calibration
                current_neutral = np.median(neutral_buffer)

                print(
                    f"NEUTRAL -> LEFT | "
                    f"peak={signed_peak:.2f})",
                    # f"(neutral={current_neutral:.2f})",
                    flush=True,
                )

                control_outlet.push_sample(["LEFT"])
                gaze_state = GAZE_LEFT # set new state and locks in even once back to neutral
                samples_in_away_state = 0
                continue

            if right_crossed: #if live signal matches right pattern from calibration
                current_neutral = np.median(neutral_buffer)

                print(
                    f"NEUTRAL -> RIGHT | "
                    f"peak={signed_peak:.2f})",
                    # f"(neutral={current_neutral:.2f})",
                    flush=True,
                )

                control_outlet.push_sample(["RIGHT"])
                gaze_state = GAZE_RIGHT #new state
                samples_in_away_state = 0
                continue

            # Safe neutral sample: adapt rolling baseline.
            neutral_buffer.append(value)
            continue


# LEFT:
# Ignore ALL detections while the user returns to
# center. After the hold period, automatically go
# back to NEUTRAL.
# ====================================================

        if gaze_state == GAZE_LEFT:

            samples_in_away_state += 1 # each sample counts time, detector ignores other detections 

            if samples_in_away_state >= GAZE_HOLD_SAMPLES: # go back to neutral after this time

                print( "Back to NEUTRAL",
                    flush=True,
                )

                #resets state
                gaze_state = GAZE_NEUTRAL
                samples_in_away_state = 0

                # Remove the LEFT movement and return-to-center
                # transient from the onset history.
                onset_buffer.clear()

                # Briefly allow the signal to settle once neutral.
                neutral_settle_counter = RETURN_SETTLE_SAMPLES

            # this ignores right_crossed, left_crossed, and all other
            # detections while we are in the LEFT state.
            continue

# ====================================================
# RIGHT:
# Ignore ALL detections while the user returns to
# center. After the hold period, automatically go
# back to NEUTRAL.
# ====================================================

        if gaze_state == GAZE_RIGHT:

            samples_in_away_state += 1

            if samples_in_away_state >= GAZE_HOLD_SAMPLES:

                print( "Back to NEUTRAL",
                    # "RIGHT -> NEUTRAL | "
                    # "hold period complete",
                    flush=True,
                )

                gaze_state = GAZE_NEUTRAL
                samples_in_away_state = 0

                # Remove the RIGHT movement and return-to-center
                # transient from the onset history.
                onset_buffer.clear()

                neutral_settle_counter = RETURN_SETTLE_SAMPLES

            # Ignore every detection while in RIGHT.
            continue

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Full program order:

        1. Start idun_pipe.exe.
        2. Find IDUN EEG stream using by starting this script.
        3. Calibrate LEFT / RIGHT (multiple trials each).
        4. Start continuous live detector.
    """

    control_outlet = (
        create_control_outlet()
    )


    inlet = find_eeg_inlet(
        wait_time=5.0
    )

    calibration, hp_filter = (
        calibrate(inlet)
    )

    run_detector(
        inlet=inlet,
        control_outlet=control_outlet,
        calibration=calibration,
        hp_filter=hp_filter,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
