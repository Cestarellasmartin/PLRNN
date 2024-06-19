# General Scripts

Each folder contains independent sripts for different objectives.

## LFP exploration

Language: Python (Jupyter)

Main script: LFP_exploration / LFP_gambling

Environtment: LFP_analysis.yml

The basic points of this script is to explore/analyse  the LFP of a specific channel. 
- LFP_exploration is focused on Claudia's task (rule switching). 
- LFP_gambling is focused on Hugo's task (gambling).

The main analysis are:
- The Power Spectrum for the full session and for specific scenearios(Correct and Incorrent trials for rule 1 and 2).
- Wavelet for the session and exploration around the switch of the rule.

## Phenosys Preprocess

Language: Matlab

Main script: Klausberger_organisation

Organization of the recording data from Phenosys system. 

The needed files:
- 'amplifier.dat'
- 'time.dat'
- 'digitalin.dat
- Files from kilosort: params.py, spike_times.npy, spike_temaplates.npy, spike_clusters.npy, amplitudes.npy, pc_features.npy, pc_features_id.npy, channel_positions.npy, templates.npy, whitening_mat_inv.npy, cluster_groups.csv, cluster_group.tsv

The output files generated are:
- Spike activity ("Filename_SpikeActivity.mat"): Spike time for each neuron in sampling points. Rows -> Times, Columns -> Neurons.
- Spike parameters ("Filename_Spike_features.mat"): Waveform parameters for each cluster / neuron.
- Wheel movement ("Filename_Wheel.mat): movement of the wheel reseting the movement to 0 in each trial.
- Metadata ("Filename_Metadata.mat"): Information about the experiment and the data collection.
## Spike Holes Detection

Language: Matlab

Main script:  Spike_hole_detection.mat

The information of this script is described in the file Readme of its folder.

