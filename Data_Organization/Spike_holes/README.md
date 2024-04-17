# Detecting "Spike Holes"

As raw electrophysiology data is heavy, the spike sorting algorithm (Kilosort) truncates the data into smaller portions (called batches) to work with it.

To ensure information continuity, the edges of these batches are processed slightly differently than the middle portion of the batch. They found that at the edge of the spike sorting batches, ~7ms of spikes are going undetected. We call this effect "spike holes".

### Method

### How to use it

### Example