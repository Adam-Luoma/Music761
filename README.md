Music761 File Notes

ARIA > folder containing last model checkpoints on ABBA train set (for use in the application)

Examples > some midi recordings with different models and video of the application

MIDI_Processing > ABBA_Melodies > Extracted melody lines from freemidi.org ABBA collection

MIDI_Processing > MIDI_Preprocess_Magenta.ipynb transposes midi into all 12 keys and saves as note sequence for Magenta Training

MIDI_Processing > MIDI_Preprocess_Markov.py builds Markov transition matrix from midi data. Normalizes pitch, tempo, and dynamics

Model Explorations > RNN_music_generation.ipynb magenta's demo script

Model Explorations > Melody_RNN.ipynb testing the use of performance RNN in magenta

Model Explorations > RNN_w_RL.ipynb magenta's RNN generation with a reinforcement learning agent integration

Model Explorations > RL_Tuner_MelodyRNN.ipynb using my trained model, generate multiple melodies and pick the best option based on reinforcement learning guided rules

BCI_Music_Frontend.py frontend application for making music. Calls the ARIA checkpoint for the most recent model, will need to update on local system path

Magenta_Train_Instructions.txt txt document containing all command line prompts to directly train magenta model with dataset acquired
