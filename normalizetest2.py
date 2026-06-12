"""
AI Melody Composer - Pygame Game
ABABCB song structure builder using fine-tuned Magenta attention_rnn

Controls:
  LEFT ARROW  — select Melody A (left option)
  RIGHT ARROW — select Melody B (right option)
  SPACE       — replay current melodies
  ESC         — quit
"""
from asyncio import streams

from pylsl import StreamInlet, resolve_streams
#from idun_guardian_sdk import GuardianClient
import pygame
import pygame.midi
import subprocess
import os
import sys
import threading
import time
import struct
import random
import math
from pathlib import Path
from scipy.signal import butter, lfilter, lfilter_zi #- normalizing time-series data
import numpy as np

RECORDING_TIMER = 0

#events needed becausepygame doesn't allow directly calling functions from another thread (eye tracker thread)- will avoid crash
SELECT_EVENT = pygame.USEREVENT + 1
LEFT_EVENT = pygame.USEREVENT + 2
RIGHT_EVENT = pygame.USEREVENT + 3

# ─────────────────────────────────────────────
#  CONFIG — edit these to match your setup
# ─────────────────────────────────────────────
#CHECKPOINT_DIR   = r"C:/Users/adamc/MusicGenAI/melody_rnn_finetuned/ABBA_05_03_26/melody_rnn/logdir/run1" #FIND PATH
CHECKPOINT_DIR   = r"C:/Users/179is/BCI-Music Project/Music761/ARIA/Checkpoints/ABBA/model.ckpt-11228.data-00000-of-00001" #FIND PATH
OUTPUT_DIR       = r"C:/Users/179is/BCI-Music_Project/Generated_Melodies" #MAKE PATH
HPARAMS          = "batch_size=64,rnn_layer_sizes=[64,64]"
NUM_STEPS_R1     = 128        # ~20 seconds at 120bpm, 4/4
NUM_STEPS_R2     = 256        # ~20 seconds at 120bpm, 4/4
TEMPERATURE      = 1.0
MIDI_INSTRUMENT  = 0          # 0 = Grand Piano

# Song structure: each tuple is (section_label, display_name, slot_in_song)
# ABABCB = A, A, B, B, C, B  — we build A, B, C independently then arrange
SONG_STRUCTURE = ["A", "B", "A", "B", "C", "B"]
SECTIONS = ["A", "B", "C"]   # unique sections to compose

# ─────────────────────────────────────────────
#  MIDI PARSING
# ─────────────────────────────────────────────

def parse_midi(path):
    """Parse a MIDI file and return list of (start_sec, duration_sec, pitch, velocity)."""
    with open(path, "rb") as f:
        data = f.read()

    pos = 0
    header = data[pos:pos+14]
    ticks_per_beat = struct.unpack(">H", header[12:14])[0]
    pos += 14

    tempo = 500000  # default 120 BPM
    notes = []

    while pos < len(data):
        chunk_id   = data[pos:pos+4]
        chunk_len  = struct.unpack(">I", data[pos+4:pos+8])[0]
        track_data = data[pos+8 : pos+8+chunk_len]
        pos       += 8 + chunk_len

        if chunk_id != b"MTrk":
            continue

        i = 0
        tick = 0
        active = {}   # pitch -> (start_tick, velocity)

        def read_varlen(buf, idx):
            val = 0
            while True:
                b = buf[idx]; idx += 1
                val = (val << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            return val, idx

        last_status = None
        while i < len(track_data):
            delta, i = read_varlen(track_data, i)
            tick += delta

            if i >= len(track_data):
                break

            b = track_data[i]

            # Running status
            if b & 0x80:
                status = b; i += 1
                last_status = status
            else:
                status = last_status

            if status is None:
                break

            ev = status & 0xF0

            if status == 0xFF:   # Meta
                meta_type = track_data[i]; i += 1
                mlen, i = read_varlen(track_data, i)
                meta_data = track_data[i:i+mlen]; i += mlen
                if meta_type == 0x51 and mlen >= 3:
                    tempo = struct.unpack(">I", b"\x00" + meta_data[:3])[0]

            elif ev == 0x90:     # Note on
                pitch = track_data[i]; i += 1
                vel   = track_data[i]; i += 1
                if vel > 0:
                    active[pitch] = (tick, vel)
                else:            # vel=0 treated as note off
                    if pitch in active:
                        st, v = active.pop(pitch)
                        sec_per_tick = (tempo / 1_000_000) / ticks_per_beat
                        notes.append((st * sec_per_tick,
                                      (tick - st) * sec_per_tick,
                                      pitch, v))

            elif ev == 0x80:     # Note off
                pitch = track_data[i]; i += 1
                _vel  = track_data[i]; i += 1
                if pitch in active:
                    st, v = active.pop(pitch)
                    sec_per_tick = (tempo / 1_000_000) / ticks_per_beat
                    notes.append((st * sec_per_tick,
                                  (tick - st) * sec_per_tick,
                                  pitch, v))

            elif ev in (0xA0, 0xB0, 0xE0):
                i += 2
            elif ev in (0xC0, 0xD0):
                i += 1

    notes.sort(key=lambda n: n[0])
    return notes


# ─────────────────────────────────────────────
#  MAGENTA GENERATION (subprocess)
# ─────────────────────────────────────────────

def _find_melody_rnn_generate():
    """Locate the melody_rnn_generate executable relative to the current Python interpreter."""
    scripts_dir = Path(sys.executable).parent
    # On Windows the entry-point lives in Scripts\; on Unix it's alongside the interpreter
    for candidate in [
        scripts_dir / "melody_rnn_generate.exe",
        scripts_dir / "melody_rnn_generate",
        scripts_dir / "Scripts" / "melody_rnn_generate.exe",
        scripts_dir / "Scripts" / "melody_rnn_generate",
    ]:
        if candidate.exists():
            return str(candidate)
    return "melody_rnn_generate"  # last-resort: rely on PATH


def generate_melodies(num_outputs=2, primer_midi_path=None, output_subdir=None, num_steps=128):
    """
    Call melody_rnn_generate via subprocess.
    Returns list of paths to generated MIDI files.
    """
    out_dir = output_subdir or os.path.join(OUTPUT_DIR, f"gen_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        _find_melody_rnn_generate(),
        "--config=attention_rnn",
        f"--run_dir={CHECKPOINT_DIR}",
        f"--output_dir={out_dir}",
        f"--hparams={HPARAMS}",
        f"--num_outputs={num_outputs}",
        f"--num_steps={num_steps}",
        f"--temperature={TEMPERATURE}",
    ]

    if primer_midi_path and os.path.exists(primer_midi_path):
        cmd.append(f"--primer_midi={primer_midi_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print("Generation error:", result.stderr)
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Could not run melody_rnn_generate: {e}")
        print("→ Using mock melodies for development")
        return _mock_generate(num_outputs, out_dir)

    midis = sorted(Path(out_dir).glob("*.mid"))
    return [str(p) for p in midis[:num_outputs]]


def _mock_generate(num_outputs, out_dir):
    """Generate simple mock MIDI files for development/testing."""
    paths = []
    scales = {
        "A": [60, 62, 64, 65, 67, 69, 71, 72],  # C major
        "B": [60, 62, 63, 65, 67, 68, 70, 72],  # C minor
    }
    for i in range(num_outputs):
        path = os.path.join(out_dir, f"mock_{i}.mid")
        scale = scales["A"] if i == 0 else scales["B"]
        _write_mock_midi(path, scale, seed=i + int(time.time()) % 1000)
        paths.append(path)
    return paths


def _write_mock_midi(path, scale, seed=0):
    """Write a simple procedural MIDI melody."""
    random.seed(seed)
    ticks_per_beat = 220
    tempo = 500000  # 120 BPM
    notes = []
    tick = 0
    durations = [110, 110, 220, 220, 440]  # 8th, 8th, quarter, quarter, half

    for _ in range(16):
        pitch = random.choice(scale) + random.choice([0, 12, -12]) 
        pitch = max(48, min(84, pitch))
        dur = random.choice(durations)
        gap = random.choice([0, 55, 110])
        notes.append((tick, pitch, dur, random.randint(60, 100)))
        tick += dur + gap

    # Build track bytes
    def varlen(n):
        if n < 128:
            return bytes([n])
        parts = []
        while n:
            parts.append(n & 0x7F)
            n >>= 7
        parts.reverse()
        return bytes([(b | 0x80) if i < len(parts)-1 else b
                      for i, b in enumerate(parts)])

    track_events = []
    # Tempo meta
    track_events.append(b"\x00\xFF\x51\x03" +
                        struct.pack(">I", tempo)[1:])

    # Note events
    all_events = []
    for (start, pitch, dur, vel) in notes:
        all_events.append((start, 0x90, pitch, vel))
        all_events.append((start + dur, 0x80, pitch, 0))
    all_events.sort()

    prev_tick = 0
    for (tick, status, pitch, vel) in all_events:
        delta = tick - prev_tick
        prev_tick = tick
        track_events.append(varlen(delta) + bytes([status, pitch, vel]))

    track_events.append(b"\x00\xFF\x2F\x00")  # End of track

    track_data = b"".join(track_events)
    track_chunk = b"MTrk" + struct.pack(">I", len(track_data)) + track_data

    header = (b"MThd" + struct.pack(">I", 6) +
              struct.pack(">HHH", 1, 2, ticks_per_beat))

    # Empty tempo track
    tempo_track_data = b"\x00\xFF\x51\x03" + struct.pack(">I", tempo)[1:] + b"\x00\xFF\x2F\x00"
    tempo_chunk = b"MTrk" + struct.pack(">I", len(tempo_track_data)) + tempo_track_data

    with open(path, "wb") as f:
        f.write(header + tempo_chunk + track_chunk)


# ─────────────────────────────────────────────
#  MIDI PLAYBACK via pygame.midi
# ─────────────────────────────────────────────

class MidiPlayer:
    def __init__(self):
        pygame.midi.init()
        self.port = pygame.midi.get_default_output_id()
        self.player = pygame.midi.Output(self.port)
        self.player.set_instrument(MIDI_INSTRUMENT)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        with self._lock:
            for pitch in range(128):
                self.player.note_off(pitch, 0)

    def play(self, notes, on_done=None):
        """Play a list of (start_sec, dur_sec, pitch, vel) notes."""
        self.stop()
        self._stop_event.clear()

        def _play():
            if not notes:
                return
            start_wall = time.perf_counter()
            # Schedule all note-on / note-off events
            events = []
            for (start, dur, pitch, vel) in notes:
                events.append((start, True,  pitch, vel))
                events.append((start + dur, False, pitch, vel))
            events.sort()

            ei = 0
            while ei < len(events) and not self._stop_event.is_set():
                ev_time, is_on, pitch, vel = events[ei]
                now = time.perf_counter() - start_wall
                wait = ev_time - now
                if wait > 0:
                    self._stop_event.wait(timeout=wait)
                if self._stop_event.is_set():
                    break
                with self._lock:
                    if is_on:
                        self.player.note_on(pitch, vel)
                    else:
                        self.player.note_off(pitch, vel)
                ei += 1

            if not self._stop_event.is_set() and on_done:
                on_done()

        self._thread = threading.Thread(target=_play, daemon=True)
        self._thread.start()

    def play_loop(self, notes_a, notes_b):
        """Alternate between two melodies continuously."""
        self.stop()
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                for notes in [notes_a, notes_b]:
                    if self._stop_event.is_set():
                        break
                    self.play(notes)
                    # wait for this melody to finish
                    if notes:
                        duration = max(s + d for s, d, *_ in notes) + 0.5
                        self._stop_event.wait(timeout=duration)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def close(self):
        self.stop()
        self.player.close()
        pygame.midi.quit()


# ─────────────────────────────────────────────
#  GAME STATE
# ─────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.sections = SECTIONS                # ["A", "B", "C"]
        self.song_structure = SONG_STRUCTURE    # ["A","B","A","B","C","B"]

        # For each section, chosen melody path + notes
        self.chosen = {s: {"path": None, "notes": None} for s in self.sections}

        # Current UI state
        self.current_section_idx = 0   # index into SECTIONS
        self.round = 1                  # 1=first pick, 2=second pick (uses primer)
        self.options = [None, None]     # two MidiOption objects
        self.status = "generating"     # generating | choosing | done
        self.playing_idx = None        # which option is playing (0 or 1 or None)
        self.generation_thread = None
        self.error = None

    @property
    def current_section(self):
        return self.sections[self.current_section_idx]

    def is_done(self):
        return all(self.chosen[s]["path"] is not None for s in self.sections)


class MidiOption:
    def __init__(self, path, notes):
        self.path = path
        self.notes = notes
        self.duration = max((s + d for s, d, *_ in notes), default=0) if notes else 0


# ─────────────────────────────────────────────
#  VISUALIZER — waveform bars
# ─────────────────────────────────────────────

def notes_to_bars(notes, n_bars=40, width_px=300):
    """Convert notes to a list of (height_fraction, pitch_norm) for bar display."""
    if not notes:
        return [(0.1, 0.5)] * n_bars

    duration = max(s + d for s, d, *_ in notes)
    bars = []
    for i in range(n_bars):
        t_start = duration * i / n_bars
        t_end   = duration * (i + 1) / n_bars
        # Find notes active in this window
        active = [n for n in notes if n[0] < t_end and n[0] + n[1] > t_start]
        if active:
            avg_pitch = sum(n[2] for n in active) / len(active)
            avg_vel   = sum(n[3] for n in active) / len(active)
            h = 0.2 + 0.8 * (avg_vel / 127)
            p = (avg_pitch - 48) / 48
        else:
            h = 0.05
            p = 0.5
        bars.append((h, max(0, min(1, p))))
    return bars


# ─────────────────────────────────────────────
#  COLORS & FONTS
# ─────────────────────────────────────────────

BG         = (10, 10, 18)
PANEL_BG   = (20, 22, 35)
PANEL_EDGE = (40, 44, 70)
ACCENT_A   = (80, 200, 255)
ACCENT_B   = (255, 120, 200)
GOLD       = (255, 200, 80)
WHITE      = (240, 240, 255)
GREY       = (120, 125, 150)
DARK_GREY  = (50, 55, 75)
GREEN      = (80, 220, 130)
RED        = (220, 80, 100)

SECTION_COLORS = {
    "A": (80, 180, 255),
    "B": (180, 100, 255),
    "C": (80, 230, 160),
}


def lerp_color(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ─────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────

W, H = 1100, 700

def draw_roundrect(surf, color, rect, radius=12, border=0, border_color=None):
    x, y, w, h = rect
    pygame.draw.rect(surf, color, (x + radius, y, w - 2*radius, h))
    pygame.draw.rect(surf, color, (x, y + radius, w, h - 2*radius))
    for cx, cy in [(x+radius, y+radius), (x+w-radius, y+radius),
                   (x+radius, y+h-radius), (x+w-radius, y+h-radius)]:
        pygame.draw.circle(surf, color, (cx, cy), radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


def draw_visualizer(surf, bars, rect, color, playing=False, phase=0.0):
    x, y, w, h = rect
    n = len(bars)
    bar_w = w / n
    for i, (height_frac, pitch_norm) in enumerate(bars):
        bx = x + i * bar_w
        bh = height_frac * h
        by = y + h - bh

        # Animate if playing
        if playing:
            wave = 0.15 * math.sin(phase * 6 + i * 0.4)
            bh = max(4, bh * (1 + wave))
            by = y + h - bh

        # Color gradient by pitch
        c = lerp_color(color, WHITE, pitch_norm * 0.3)
        alpha_surf = pygame.Surface((max(1, int(bar_w) - 1), max(1, int(bh))), pygame.SRCALPHA)
        alpha_surf.fill((*c, 200 if playing else 140))
        surf.blit(alpha_surf, (int(bx), int(by)))


def draw_song_map(surf, font_sm, state, x, y, w):
    """Draw the ABABCB song map at the bottom."""
    section_w = w // len(state.song_structure)
    for i, sec in enumerate(state.song_structure):
        sx = x + i * section_w
        color = SECTION_COLORS[sec]
        chosen = state.chosen[sec]["path"] is not None

        if chosen:
            draw_roundrect(surf, color, (sx+2, y, section_w-4, 36), radius=8)
            label = font_sm.render(sec, True, BG)
        else:
            draw_roundrect(surf, DARK_GREY, (sx+2, y, section_w-4, 36), radius=8)
            label = font_sm.render(sec, True, GREY)

        surf.blit(label, (sx + section_w//2 - label.get_width()//2,
                          y + 18 - label.get_height()//2))


def draw_spinner(surf, cx, cy, t, color=ACCENT_A, r=28):
    for i in range(8):
        angle = (t * 3 + i * math.pi / 4) % (2 * math.pi)
        alpha = int(255 * (i / 8))
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        pygame.draw.circle(surf, (*color, alpha), (int(px), int(py)), 4)




#____________________________________________
#   EYE CONTROLS
#____________________________________________

# for left eye movement 
def left_move(state, midi):
    """move to left option"""
    #event.key == pygame.K_LEFT
    #if state.status == "choosing":
    midi.play(state.options[0].notes)
    state.playing_idx = 0

#for right eye movement
def right_move(state, midi): 
    """move to right option"""
    #event.key == pygame.K_RIGHT
    midi.play(state.options[1].notes)
    state.playing_idx = 1


#____________________________________________
#   CALIBRATION
#____________________________________________

#to use between left and right looking
def countdown(seconds=3):
    for i in range (seconds, 0, -1):
        print(f"{i}...", flush=True)
        time.sleep(1)

def calibrate(inlet, num_samples):
    
    print('Welcome to BCI-Music! First we will calibrate your right and left eye movements to control the game', flush=True)
    time.sleep(4)

    #for normalization, collect neutral baseline dat
    print('First, look straight ahead to establish baseline', flush=True)
    countdown(5) #countdown to give user time to get ready and look straight ahead for neutral baseline collection
    while True: #buffer is full of old data, so we pull with timeout until it's empty to start fresh
        sample, ts = inlet.pull_sample(timeout=0.0) #pull sample with timeout=0 so we can break out of loop when buffer is empty
        if sample is None: #if no sample is returned, buffer is empty, so we can break
            break
    print("Stay neutral...", flush=True)
    neutral_samples = [] #collect neutral samples to calculate baseline
    for _ in range(num_samples): # this is where we collect the neutral baseline data to normalize against for the left and right samples
        neutral_samples.append(sample[0])
    neutral_baseline = np.median(neutral_samples) #use median to reduce impact of outliers/spikes in the signal but have also tried mean

    # drain buffer because there may be some samples collected while we were processing the neutral baseline, so we pull with timeout until it's empty to start fresh for left calibration
    while True:
        sample, ts = inlet.pull_sample(timeout=0.0)
        if sample is None:
            break

    # this is where we collect the left and right samples to find the peaks for each eye movement, which we will then normalize against the neutral baseline to get the actual movement signal that we can use for detection in the game
    print('Left calibration beginning in', flush=True)
    countdown(5)
    while True:
        sample, ts = inlet.pull_sample(timeout=0.0)
        if sample is None:
            break
    print("Calibration: look LEFT now...", flush=True)
    left_samples = []
    for _ in range(num_samples):
        sample, _ = inlet.pull_sample()
        left_samples.append(sample[0])
    print('Left calibration complete.\n', flush=True)

    # drain buffer
    while True:
        sample, ts = inlet.pull_sample(timeout=0.0)
        if sample is None:
            break

    # for right eye movement calibration, same process as left 
    print('Right calibration beginning in', flush=True)
    countdown(5)
    while True:
        sample, ts = inlet.pull_sample(timeout=0.0)
        if sample is None:
            break
    print("Calibration: look RIGHT now...", flush=True)
    right_samples = []
    for _ in range(num_samples):
        sample, _ = inlet.pull_sample()
        right_samples.append(sample[0])
    print('Right calibration complete.\n', flush=True)

    # take medians of collected samples then subtract the neutral baseline to get the actual movement peaks for left and right, which we will use for detection thresholds in the game
    left_peak  = np.median(left_samples)  - neutral_baseline
    right_peak = np.median(right_samples) - neutral_baseline
    # threshold  = abs(left_peak - right_peak) * 0.7

    print(f"Neutral baseline:       {neutral_baseline:.2f}", flush=True)
    print(f"Left  (normalized):     {left_peak:.2f}", flush=True)
    print(f"Right (normalized):     {right_peak:.2f}", flush=True)
    # print(f"Threshold set to:       {threshold:.2f}", flush=True)
    time.sleep(2)
    print('Begin game play in...', flush=True)
    countdown(5)

    return left_peak, right_peak



def start_eye_tracker(inlet, left_peak, right_peak):
    ''' this function runs in a separate thread and continuously pulls samples from the LSL stream, 
    applies the high pass filter, normalizes against the baseline, and checks for eye movement detections 
    based on the calibrated peaks and thresholds. 
    When it detects a left or right eye movement, it posts a 
    pygame event that the main game loop can pick up to trigger the corresponding action.'''


    print("EEG stream found!\n", flush=True) #after eye tracker is called in main function and stream is already resolved

    COOLDOWN = 1250  # 5 seconds at 250Hz before next detection
    cooldown_counter = 0 #this counter is set to COOLDOWN when a detection occurs, and counts down each sample until it reaches 0, at which point detection can occur again. 
    #This prevents multiple detections from a single eye movement due to signal noise or holding the movement for too long.

    # Set up high pass filter - removed because wasn't working but can try again
    # removes slow baseline drift from the signal
    # def high_pass_filter(cutoff=0.5, fs=250, order=4):
    #     nyq = fs / 2
    #     normal_cutoff = cutoff / nyq
    #     b, a = butter(order, normal_cutoff, btype='high', analog=False)
    #     zi = lfilter_zi(b, a)
    #     return b, a, zi

    # b, a, zi = high_pass_filter()
    from collections import deque #deque is a double-ended queue that we can use as a rolling buffer to store the most recent samples for baseline calculation.
    # We set a maxlen so it automatically discards old samples when new ones come in, which is good for real-time processing needs.

    BASELINE_WINDOW = 750  # 3 seconds at 250Hz #this is the number of recent samples we keep to calculate a rolling baseline for normalization
    baseline_buffer = deque(maxlen=BASELINE_WINDOW) # this buffer will hold the most recent samples to calculate a rolling baseline, which helps account for slow changes in the signal over time
    #keeps the detection responsive to actual eye movements rather than drift.

    # fill baseline before detection
    print("Collecting baseline...", flush=True)
    for _ in range(BASELINE_WINDOW):
        sample, _ = inlet.pull_sample()
        baseline_buffer.append(sample[0])
    print("Baseline collected!\n", flush=True)

    # ---- Settle the filter ----
    # the filter produces a large spike on the first few samples
    # so we run 500 samples through it (2 seconds) before detection
    # to let it stabilize
    # print("Settling filter...", flush=True)
    # for _ in range(500):
    #     sample, _ = inlet.pull_sample()
    #     _, zi = lfilter(b, a, [sample[0]], zi=zi)
    # print("Ready!\n", flush=True)

    # calculate detection thresholds based on the calibrated peaks for left and right eye movements.
    # we use 70% of each peak as the detection threshold so the user doesn't need to make a full eye movement to trigger
    left_threshold  = left_peak  * 0.7
    right_threshold = right_peak * 0.7

    # determine detection direction for each threshold since in brainviewer L and R are different than plotted raw eeg seen
    # left/right peaks can be positive or negative, so we check which direction each peak went and set the detection accordingly
    # if threshold is negative, signal needs to go below it
    # if threshold is positive, signal needs to go above it
    if left_threshold < 0:
        left_direction = "below"
    else:
        left_direction = "above"

    if right_threshold < 0:
        right_direction = "below"
    else:
        right_direction = "above"

    print(f"Left threshold:  {left_threshold:.2f} (detect when signal goes {left_direction})", flush=True)
    print(f"Right threshold: {right_threshold:.2f} (detect when signal goes {right_direction})", flush=True)
    print("Starting detection...", flush=True)

    # main detection loop with high pass filter and no rolling buffer
    # while True:
    #     # pull one sample from the LSL stream
    #     sample, _ = inlet.pull_sample()
    #     value = sample[0]

    #     # apply high pass filter to remove slow drift
    #     # zi carries the filter state between samples so it works in real time
    #     corrected, zi = lfilter(b, a, [value], zi=zi)
    #     corrected = corrected[0]
    # then in detection loop replace the filter lines with:

    while True:
        sample, _ = inlet.pull_sample() #this reacts to each new piece of data as it comes in rather than waiting for a full buffer to fill up
        value = sample[0]

        baseline = np.mean(baseline_buffer) #calculate the current baseline from the rolling buffer of recent samples, which helps account for slow changes in the signal over time
        corrected = value - baseline
        if cooldown_counter > 0: # if we're in cooldown period after a detection, we skip detection and just update the baseline buffer until cooldown is over
                    cooldown_counter -= 1
                    continue


        baseline_buffer.append(value) # add the new sample to the baseline buffer for future baseline calculations, good for eye movement and not all drift

        # ---- Detect left eye movement ----
        # check above or below threshold depending on which direction
        # left peak went during calibration
        if left_direction == "below":
            left_detected = corrected < left_threshold
        else:
            left_detected = corrected > left_threshold

        # ---- Detect right eye movement ----
        # same logic for right
        if right_direction == "below":
            right_detected = corrected < right_threshold
        else:
            right_detected = corrected > right_threshold

        # post pygame events instead of calling functions directly
        # because pygame functions can only be called from the main thread
        # the main game loop picks these up and calls left_move/right_move safely
        if left_detected:
            print("Left eye movement detected", flush=True)
            pygame.event.post(pygame.event.Event(LEFT_EVENT))
            cooldown_counter = COOLDOWN

        elif right_detected:
            print("Right eye movement detected", flush=True)
            pygame.event.post(pygame.event.Event(RIGHT_EVENT))
            cooldown_counter = COOLDOWN

# ─────────────────────────────────────────────
#  MAIN GAME
# ─────────────────────────────────────────────

def main():

    streams = resolve_streams(wait_time=5.0) #wait for up to 5 seconds to find an LSL stream. If no stream is found after 5 seconds, it will raise an error and exit, since the game relies on this data to function.
    inlet = StreamInlet(streams[0]) #connect to first stream available
    left_peak, right_peak = calibrate(inlet, num_samples= 1250) #5 seconds of listening at 250Hz to collect the samples for calibration
   

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("AI Melody Composer — ABABCB")
    clock = pygame.time.Clock()

    # Fonts
    try:
        font_lg = pygame.font.SysFont("Georgia", 36, bold=True)
        font_md = pygame.font.SysFont("Georgia", 22)
        font_sm = pygame.font.SysFont("Georgia", 16)
        font_xs = pygame.font.SysFont("Courier New", 13)
    except:
        font_lg = pygame.font.Font(None, 40)
        font_md = pygame.font.Font(None, 26)
        font_sm = pygame.font.Font(None, 20)
        font_xs = pygame.font.Font(None, 16)

    midi = MidiPlayer()
    state = GameState()


    # Animation state
    phase = 0.0
    spinner_t = 0.0
    bars_cache = [None, None]
    feedback_msg = ""
    feedback_timer = 0
    final_screen = False


    def start_generation():
        state.status = "generating"
        state.options = [None, None]
        bars_cache[0] = None
        bars_cache[1] = None
        midi.stop()

        steps = NUM_STEPS_R1 if state.round == 1 else NUM_STEPS_R2
        primer = None
        if state.round == 2 and state.chosen[state.current_section]["path"]:
            primer = state.chosen[state.current_section]["path"]

        def gen_thread():
            paths = generate_melodies(num_outputs=2, primer_midi_path=primer, num_steps=steps)
            opts = []
            for p in paths:
                try:
                    notes = parse_midi(p)
                    opts.append(MidiOption(p, notes))
                except Exception as e:
                    print(f"Failed to parse {p}: {e}")

            if len(opts) < 2:
                state.error = "Generation failed — check Magenta install"
                state.status = "error"
                return

            state.options[0] = opts[0]
            state.options[1] = opts[1]
            bars_cache[0] = notes_to_bars(opts[0].notes)
            bars_cache[1] = notes_to_bars(opts[1].notes)
            state.status = "choosing"
            # Auto-play left option first
            midi.play(opts[0].notes)
            state.playing_idx = 0

        state.generation_thread = threading.Thread(target=gen_thread, daemon=True)
        state.generation_thread.start()

    def select_option(idx):
        nonlocal feedback_msg, feedback_timer
        opt = state.options[idx]
        if opt is None:
            return

        midi.stop()
        sec = state.current_section

        if state.round == 1:
            # First pick for this section — store as primer, go to round 2
            state.chosen[sec] = {"path": opt.path, "notes": opt.notes}
            state.round = 2
            feedback_msg = f"Section {sec} primer set — now refining..."
            feedback_timer = 90
            start_generation()

        else:
            # Second pick — this is the FINAL choice for this section
            state.chosen[sec] = {"path": opt.path, "notes": opt.notes}
            state.round = 1
            state.current_section_idx += 1
            feedback_msg = f"Section {sec} locked in ✓"
            feedback_timer = 90

            if state.is_done():
                state.status = "done"
                play_full_song()
            else:
                start_generation()

    #this thread runs the eye tracker in parallel to the main game loop, continuously listening for eye movement detections 
    # and posting events when they occur, so the main loop can respond to them without blocking or missing any detections.
    idun_thread = threading.Thread(
        target=start_eye_tracker,
        args=(inlet,left_peak, right_peak),
        daemon=True
        )
    idun_thread.start()

    def play_full_song():
        nonlocal final_screen
        final_screen = True
        # Build full song note sequence
        all_notes = []
        cursor = 0.0
        for sec in state.song_structure:
            notes = state.chosen[sec]["notes"]
            if notes:
                dur = max(s + d for s, d, *_ in notes)
                shifted = [(s + cursor, d, p, v) for s, d, p, v in notes]
                all_notes.extend(shifted)
                cursor += dur + 0.5  # small gap between sections
        midi.play(all_notes)
    
    def replay():
        if state.status == "choosing":
            idx = state.playing_idx or 0
            if state.options[idx]:
                midi.play(state.options[idx].notes)

    # Kick off first generation
    start_generation()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        phase += dt
        spinner_t += dt
        if feedback_timer > 0:
            feedback_timer -= 1

        # ── Events ──────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == LEFT_EVENT: #adding as event type because pygame doesn't allow directly calling functions from another thread (eye tracker thread)
                    if state.status == "choosing":
                        left_move(state, midi)

            elif event.type == RIGHT_EVENT: #adding as event type because pygame doesn't allow directly calling functions from another thread (eye tracker thread)
                if state.status == "choosing":
                    right_move(state, midi)

            elif event.type == SELECT_EVENT:
                if state.status == "choosing":
                    select_option(event.idx)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif state.status == "choosing":
                    #pred_handler(event, state, midi)
                    if event.key == pygame.K_LEFT:
                        left_move(state, midi)
                    #     midi.play(state.options[0].notes)
                    #     state.playing_idx = 0
                    elif event.key == pygame.K_RIGHT:
                        right_move(state, midi)
                        # midi.play(state.options[1].notes)
                        # state.playing_idx = 1
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_z:
                        select_option(0)
                    elif event.key == pygame.K_SLASH or event.key == pygame.K_x:
                        select_option(1)
                    elif event.key == pygame.K_SPACE:
                        replay()

                elif state.status == "done":
                    if event.key == pygame.K_SPACE:
                        play_full_song()
                    elif event.key == pygame.K_r:
                        # Restart
                        state.__init__()
                        final_screen = False
                        start_generation()

        # ── Draw ─────────────────────────────────────
        screen.fill(BG)

        # Subtle grid background
        for gx in range(0, W, 60):
            pygame.draw.line(screen, (18, 20, 32), (gx, 0), (gx, H))
        for gy in range(0, H, 60):
            pygame.draw.line(screen, (18, 20, 32), (0, gy), (W, gy))

        if final_screen:
            # ── DONE SCREEN ──
            draw_roundrect(screen, PANEL_BG, (100, 80, W-200, H-160), radius=20)
            draw_roundrect(screen, (0,0,0,0), (100, 80, W-200, H-160), radius=20,
                           border=2, border_color=GOLD)

            title = font_lg.render("🎵  Song Complete!", True, GOLD)
            screen.blit(title, (W//2 - title.get_width()//2, 120))

            sub = font_md.render("Your ABABCB composition is playing...", True, WHITE)
            screen.blit(sub, (W//2 - sub.get_width()//2, 175))

            # Song map
            sy = 240
            for i, sec in enumerate(state.song_structure):
                bx = 160 + i * 120
                color = SECTION_COLORS[sec]
                draw_roundrect(screen, color, (bx, sy, 100, 80), radius=10)
                lbl = font_lg.render(sec, True, BG)
                screen.blit(lbl, (bx + 50 - lbl.get_width()//2, sy + 40 - lbl.get_height()//2))

            # Instructions
            hint1 = font_sm.render("SPACE — replay full song", True, GREY)
            hint2 = font_sm.render("R — start over", True, GREY)
            hint3 = font_sm.render("ESC — quit", True, GREY)
            screen.blit(hint1, (W//2 - hint1.get_width()//2, H - 200))
            screen.blit(hint2, (W//2 - hint2.get_width()//2, H - 175))
            screen.blit(hint3, (W//2 - hint3.get_width()//2, H - 150))

        else:
            # ── HEADER ───────────────────────────────
            sec = state.current_section if state.current_section_idx < len(SECTIONS) else "—"
            sec_color = SECTION_COLORS.get(sec, WHITE)
            round_str = f"Round {state.round}/2"

            header_txt = font_lg.render(
                f"Section  ", True, WHITE)
            sec_badge = font_lg.render(sec, True, sec_color)
            round_badge = font_md.render(round_str, True, GREY)

            screen.blit(header_txt, (40, 28))
            screen.blit(sec_badge, (40 + header_txt.get_width(), 28))
            screen.blit(round_badge, (40 + header_txt.get_width() + sec_badge.get_width() + 16, 38))

            # Progress dots
            for i, s in enumerate(SECTIONS):
                done = state.chosen[s]["path"] is not None
                active = (i == state.current_section_idx)
                c = SECTION_COLORS[s] if (done or active) else DARK_GREY
                r = 8 if active else 6
                pygame.draw.circle(screen, c,
                    (W - 180 + i * 40, 44), r)
                if done:
                    pygame.draw.circle(screen, GREEN,
                        (W - 180 + i * 40, 44), r, 2)

            # Instruction bar
            if state.status == "choosing":
                hint = font_xs.render(
                    "← / → listen    Z select left    X select right    SPACE replay",
                    True, GREY)
                screen.blit(hint, (W//2 - hint.get_width()//2, 72))

            # ── PANELS ──────────────────────────────
            panel_w = (W - 80) // 2
            panel_h = 380

            for idx in range(2):
                px = 20 + idx * (panel_w + 20)
                py = 100

                is_playing = (state.playing_idx == idx and state.status == "choosing")
                is_primer  = (state.round == 2 and
                              state.chosen[state.current_section]["path"] ==
                              (state.options[idx].path if state.options[idx] else None))

                # Panel bg
                edge_color = ACCENT_A if idx == 0 else ACCENT_B
                edge_color = edge_color if is_playing else PANEL_EDGE
                draw_roundrect(screen, PANEL_BG, (px, py, panel_w, panel_h), radius=16)
                draw_roundrect(screen, (0,0,0,0), (px, py, panel_w, panel_h),
                               radius=16, border=2,
                               border_color=edge_color)

                # Label
                key_lbl = "Z" if idx == 0 else "X"
                arrow   = "◀  LEFT" if idx == 0 else "RIGHT  ▶"
                color   = ACCENT_A if idx == 0 else ACCENT_B
                lbl = font_md.render(f"Option {idx+1}", True, color)
                key = font_sm.render(f"{arrow}  |  {key_lbl} to select", True, GREY)
                screen.blit(lbl, (px + panel_w//2 - lbl.get_width()//2, py + 16))
                screen.blit(key, (px + panel_w//2 - key.get_width()//2, py + 46))

                # Visualizer area
                vis_rect = (px + 24, py + 80, panel_w - 48, 180)

                if state.status == "generating" or state.options[idx] is None:
                    # Spinner
                    cx = px + panel_w // 2
                    cy = py + 80 + 90
                    surf_alpha = pygame.Surface((panel_w - 48, 180), pygame.SRCALPHA)
                    surf_alpha.fill((0, 0, 0, 0))
                    draw_spinner(surf_alpha, (panel_w-48)//2, 90, spinner_t, color)
                    screen.blit(surf_alpha, (px + 24, py + 80))
                    gen_txt = font_sm.render("Generating...", True, GREY)
                    screen.blit(gen_txt, (cx - gen_txt.get_width()//2, cy + 40))

                else:
                    bars = bars_cache[idx]
                    if bars:
                        draw_visualizer(screen, bars, vis_rect, color,
                                        playing=is_playing, phase=phase)

                    # Duration label
                    dur = state.options[idx].duration
                    dur_txt = font_xs.render(f"{dur:.1f}s", True, GREY)
                    screen.blit(dur_txt, (px + panel_w - 60, py + 270))

                    # Playing indicator
                    if is_playing:
                        pip = font_sm.render("▶  PLAYING", True, color)
                        screen.blit(pip, (px + panel_w//2 - pip.get_width()//2, py + 275))

                    # Note count
                    nc = len(state.options[idx].notes)
                    nc_txt = font_xs.render(f"{nc} notes", True, DARK_GREY)
                    screen.blit(nc_txt, (px + 10, py + 270))

                # Pitch ladder (decorative)
                for j in range(7):
                    gy = py + 80 + 20 + j * 22
                    pygame.draw.line(screen, DARK_GREY,
                                     (px + 8, gy), (px + 18, gy), 1)

            # ── ERROR STATE ──────────────────────────
            if state.status == "error":
                err_surf = font_md.render(state.error or "Unknown error", True, RED)
                screen.blit(err_surf, (W//2 - err_surf.get_width()//2, 310))

            # ── FEEDBACK MESSAGE ─────────────────────
            if feedback_timer > 0:
                alpha = min(255, feedback_timer * 6)
                fb = font_sm.render(feedback_msg, True, GREEN)
                screen.blit(fb, (W//2 - fb.get_width()//2, 500))

            # ── SONG MAP ─────────────────────────────
            draw_song_map(screen, font_sm, state, 20, H - 80, W - 40)
            map_lbl = font_xs.render("SONG STRUCTURE", True, GREY)
            screen.blit(map_lbl, (20, H - 100))

            # ── ROUND EXPLANATION ────────────────────
            if state.status == "choosing":
                if state.round == 1:
                    exp = font_xs.render(
                        f"Pick your favourite melody for Section {sec}  —  it will be used as a primer for the next round",
                        True, GREY)
                else:
                    exp = font_xs.render(
                        f"Section {sec} primer locked.  Now pick the final melody (AI has used your choice as inspiration)",
                        True, GREY)
                screen.blit(exp, (W//2 - exp.get_width()//2, 498))

        pygame.display.flip()

    midi.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":

    state, midi = main()
  