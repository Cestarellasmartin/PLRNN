function [wf,chMap]=GetWaveform(Name,DataType,nCh,wfWin,nWf,Clu,Folder,SR,sp)

gwfparams.dataDir = Folder;    % KiloSort/Phy output folder
% apD = dir(fullfile(myKsDir, '*ap*.bin')); % AP band file from spikeGLX specifically
% gwfparams.fileName = apD(1).name;         % .dat file containing the raw 
gwfparams.fileName = Name;
gwfparams.dataType = DataType;            % Data type of .dat file (this should be BP filtered)
gwfparams.nCh = nCh;                      % Number of channels that were streamed to disk in .dat file
gwfparams.wfWin = wfWin;              % Number of samples before and after spiketime to include in waveform
gwfparams.nWf = nWf;                    % Number of waveforms per unit to pull out
gwfparams.spikeTimes = ceil(sp.st(sp.clu==Clu)*SR); % Vector of cluster spike times (in samples) same length as .spikeClusters
gwfparams.spikeClusters = sp.clu(sp.clu==Clu);

[wf,chMap] = getWaveForms(gwfparams);