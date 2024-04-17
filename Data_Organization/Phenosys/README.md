# README

The data of the Gambling task is organised by the script `Klausberger_organisation.m` . This scripts will generate four files:
- **XXXX_SpikeActivity:** Matrix (TxN). Each column (N) is an individual neuron and its row (T) indicates the spike time in sampling points. The length T correspond to the maximum number of spikes of the population. The rest of the rows without spikes are NaN
  
- **XXXX_SpikeParameters:** Table of the waveform properties for each neuron

- **XXXX_Wheel:** Vector. The length of the vector is the total number of Sampling Points. Each component is a degree of the wheel and it is reset when a new trial starts.

- **XXXX_Metadata:** Extra information about the experiment. Part of the information will be provided by the user (see next section).

The letters XXXX will be replaced by the name added by the user (see next section)

## How to run it
The script is thought to be added in the Matlab's path. Once they are added:

1) Execute the script `Klausberger_organisation.m`
2) Select the folder where you have the raw data (amplifier.dat, time.dat & digitalin.dat)
3) Write the information required in the terminal and the prefix (XXXX) for the files generated. It could be the label of the animal and the data, for instance (CEM01_20231005)
  
The sript will obtain, organised and generate the four explained files.

### Optional: Local Field Potential 


## Necessary functions


### WARNING:
- To write the str variable use this quote `' '` and not this `" "`. If not you will have problems with Python
- The LFP is decimate to a Sample rate of 5000 Hz and then filtered with a lowpass filter with a bandpass of 300 Hz. The information of the filter is saved, as well in the variable LFP.

## Main functions of the script

The script organizes the data in the structure explained in the document EphysStorage090421. For that there are two key functions performed:

  1- Units/Neurons are classified into "Putative pyramidal cells" and "Putative interneurons" depending on the waveform parameters. After classification you will see a scatter plot with the two clusters. If the cluster analysis indentify three clusters you will have a warning message and you will have an extra neuron called "High frequency neurons". Neurons that aren't classified correctly are labelled as "Unknown". To classify the neuron is used three matlab functions: `Spikeparameters.m`, `GetWaveform.m` and `getwaveform.m`.
  
  2- The filter used is a low passband filter to filter the signal above 300 Hz with a cutoff around 450-500 Hz. The signal also is downsampling to 5 KhZ to reduce the weight of the file.
  
## Testing

Scripts `Klausberger_organisation.m` have been tested in 8 sessions: 
- JG15_190722
- JG15_190723
- JG15_190724
- JG15_190725
- JG15_190726
- JG15_190728
- CE03_20200827
- JG18_190828
