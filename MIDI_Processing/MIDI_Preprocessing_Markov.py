import music21
import numpy as np
import os
from music21 import chord

cd = os.getcwd()

#Order of operartions to preprocess MIDI data:
#1. Load MIDI files and extract metadata (key, tempo)
#2. Normalize pitch and tempo (Cmaj/Amin, 120bpm)
#3. Normalize dynamics (V-Vmin/Vmax-Vmin)

#dictionary to hold MIDI data and metadata
scores = {
    "name": [],
    "mid": [],
    "key": [],
    "tempo": [],
    "tonic": [],
    "normalized_mid": []
}

#load in MIDI files
def load_data():
    for files in os.listdir(cd + '/BJ_Data/'):
        if files.endswith('.mid') or files.endswith('.midi'):
            file_path = os.path.join(cd + '/BJ_Data/', files)
            scores["name"].append(file_path)
            scores["mid"].append(music21.converter.parse(file_path))
            scores["normalized_mid"].append(None)  # Placeholder for normalized MIDI
    return scores   

load_data()

#define key and tempo for pitch and timing normalization
def extract_key_and_tempo(scores):
    for mid in scores["mid"]:
        # Extract key
        key = mid.analyze('key')
        scores["key"].append(str(key))
        scores["tonic"].append(key.tonic.name)
        # # Extract tempo
        tempos = mid.metronomeMarkBoundaries()
        if tempos:
            tempo = tempos[0][2].number  # Get the first tempo marking
        scores["tempo"].append(tempo)

extract_key_and_tempo(scores)

#normalizing pitch
def normalize_pitch(scores):
    for i, tonic in enumerate(scores['tonic']):   
        # Transpose to C major or A minor
        if 'major' in scores['key'][i]:
            interval = music21.interval.Interval(music21.pitch.Pitch(tonic), music21.pitch.Pitch('C'))
        else:
            interval = music21.interval.Interval(music21.pitch.Pitch(tonic), music21.pitch.Pitch('A'))
        scores['normalized_mid'][i] = scores['mid'][i].transpose(interval)
        scores['normalized_mid'][i] = scores['normalized_mid'][i].flatten(retainContainers=True)

normalize_pitch(scores)

## TO DO Tempo and dynamics normalization
# # normalizing tempo
# def normalize_tempo(scores, target_tempo=120):
#     for i, mid in enumerate(scores['normalized_mid']):
#         current_tempo = scores['tempo'][i]
#         tempo_ratio = target_tempo / current_tempo
#         for el in mid.recurse().getElementsByClass(music21.tempo.MetronomeMark):
#             el.number = target_tempo
#         for n in mid.flat.notes:
#             n.quarterLength *= tempo_ratio


# normalize_tempo(scores, target_tempo=120)

# normalize dynamics
# def normalize_dynamics(scores):
#     piece_min = []
#     piece_max = []
    
#     for i, mid in enumerate(scores['normalized_mid']):
#         volumes = [n.volume.velocity for n in mid.flat.notes if n.volume and n.volume.velocity is not None]
#         Vmin = min(volumes)
#         piece_min.append(Vmin)
#         Vmax = max(volumes)
#         piece_max.append(Vmax)
    
#     glob_min = min(piece_min)
#     glob_max = max(piece_max)
    
#     for i, mid in enumerate(scores['normalized_mid']):
#         mid_volumes = [n.volume.velocity for n in mid.flat.notes if n.volume and n.volume.velocity is not None]
    
    
    
#     if volumes:
#         Vmin = min(volumes)
#         Vmax = max(volumes)
#         for n in mid.flat.notes:
#             if n.volume and n.volume.velocity is not None:
#                 n.volume.velocity = int((n.volume.velocity - Vmin) / (Vmax - Vmin))




# def normalize_dynamics(scores):
#     for mid in scores["mid"]:
#         Vmax = argmax(mid.volume)
        
# generate transition matrix from normalized MIDI data
def generate_melody_matrix(scores):
    transition_matrix = np.zeros((128, 128))  # MIDI note range from 0 to 127

    # Flatten the normalized MIDI data by track
    for mid in scores['normalized_mid']:
        flat_mid = mid.flatten(retainContainers=True)
       
        # Index tracks and check for chords to see if harmony or melody
        for track in flat_mid.parts:
            chord_in_track = track[j].getElementsByClass(chord.Chord)
            
            # If chords are present, skip this track
            if len(chord_in_track) > 0:
                continue
            else:
                for n, tone in enumerate(track[j]):
                    notes = [n.pitch.midi for n in track[j].notes if n.isNote]
    print(notes)          
            
generate_melody_matrix(scores)


#                 # Generate transitions within this track
#                 for (note1, note2) in zip(notes[:-1], notes[1:]):
#                     transition_matrix[note1][note2] += 1

#     # Normalize the matrix
#     row_sums = transition_matrix.sum(axis=1, keepdims=True)
#     transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)

#     return transition_matrix

# transition_matrix = generate_melody_matrix(scores)
# print(np.argmax(transition_matrix))