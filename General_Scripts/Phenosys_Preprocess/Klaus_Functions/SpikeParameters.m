function [NeuronsParams, ParamStruct, Clus, Spike_Parameters] = SpikeParameters
%% Setting initial variables
%Cristian version
% Setting the number of channels
% By reading the amplifier.dat and the time.dat, it is possible to know the
% number of channles depening on the proportion of their ratios
Amp = dir('amplifier.dat');
Tim = dir('time.dat');
nChannels = round(4*Amp.bytes/(2*Tim.bytes)); %Check that the rounding works
disp(['Number of Channels: ',num2str(nChannels)])

SR = 20000;
Resample_SR = 2000000;
Interp_Magnitude = Resample_SR/SR;
Interval_Spike = 2; % Miliseconds before and after the spike detection
NumPoint = Interval_Spike/1000*SR;
myKsDir = pwd;   % Sets the path where the clustered spikes are
sp = loadKSdir(myKsDir); % Load the information of the clusters
Clus=sp.cids(sp.cgs==2); % Ids of the clusters which are labelled as good while clustering
Time = dir('time.dat');
LengthInSamples = Time.bytes/4;
%% Extracting spikes
cont = 1; %Counter for the cells
f = waitbar(0,'Waiting...');

for i = Clus
    %disp(['Cluster: ',num2str(i)])
    %disp([num2str(cont),' of ',num2str(length(Clus)),' Clusters'])
    Clu=i;
    [W.wf, W.chMap]=GetWaveform('amplifier.dat','int16',nChannels,[-NumPoint NumPoint],1000,Clu,myKsDir,SR,sp);
    W.Waveform=squeeze(W.wf.waveFormsMean);
    W.A=max(W.Waveform')-min(W.Waveform');
    W.Channel = find(W.A==max(W.A),1);
    W.Spikes=length(sp.st(sp.clu==Clu));
    
    %%
    W.Selected_Waveform = W.Waveform(W.Channel,:);
% % %     x = linspace(-Interval_Spike,Interval_Spike,length(Selected_Waveform));
% % %     plot(x,Selected_Waveform)
% % %     title(['Cluster: ',num2str(Clu),' channel: ',num2str(chMap(Channel)-1)])
    %New_Waveform = resample(Selected_Waveform,2000000,20000,'spline');
    W.New_Waveform = interp(W.Selected_Waveform,Interp_Magnitude);
% % %     x = linspace(-Interval_Spike,Interval_Spike,length(New_Waveform));
% % %     hold on
% % %     plot(x,New_Waveform)
    %% Setting points
    S.Zero = find(W.New_Waveform == min(W.New_Waveform));
    S.BaselineAmplitude = mean(W.New_Waveform(1:1000));
    % Finding peaks
    [S.pks,S.I]=findpeaks(W.New_Waveform,'MinPeakheight',W.New_Waveform(S.Zero)-S.BaselineAmplitude/2);
    S.FirstPeak = S.I(find(S.I<S.Zero,1,'last')); %First peak before the trough
    S.LastPeak = S.I(find(S.I>S.Zero,1)); %First peak after the trough
    % Neccesary amplitudes
    
    S.Peak2Trough = W.New_Waveform(S.FirstPeak)-W.New_Waveform(S.Zero);
    S.Baseline2Trough= S.BaselineAmplitude - W.New_Waveform(S.Zero);
    S.Peak2Trough75 = W.New_Waveform(S.FirstPeak)-0.75*S.Peak2Trough; % 75% of peak to trough
    S.Baseline2Trough75 = S.BaselineAmplitude-0.75*S.Baseline2Trough; % 75% of peak to trough
    
    %% Parameters
    P.FR = W.Spikes/(LengthInSamples/SR);
    P.Parameter.FR = P.FR;
    NeuronsParams{cont,1} = P.Parameter.FR;
    
    P.Parameter.Peak2Peak = (S.LastPeak - S.FirstPeak)/Resample_SR;
    NeuronsParams{cont,2} = P.Parameter.Peak2Peak;
    
    P.Parameter.Trough2LastPeak = (S.LastPeak - S.Zero)/Resample_SR;
    NeuronsParams{cont,3} = P.Parameter.Trough2LastPeak;
    
    P.Parameter.Firstpeak2Trough = (S.Zero - S.FirstPeak)/Resample_SR;
    NeuronsParams{cont,4} = P.Parameter.Firstpeak2Trough;
    
    %This neuron cannot show the spike, delayed in the spiketime or cover
    %by other signal
    if isempty(S.Peak2Trough)
        NeuronsParams{cont,1} = 0;
        NeuronsParams{cont,2} = 0;
        NeuronsParams{cont,3} = 0;
        NeuronsParams{cont,4} = 0;
        NeuronsParams{cont,5} = 0;
        NeuronsParams{cont,6} = 0;
        NeuronsParams{cont,7} = 0;
        cont = cont + 1;
        continue
    end
    P.Peak2Trough1P = find(W.New_Waveform(1:S.Zero) < S.Peak2Trough75,1);
    P.Peak2Trough2P = S.Zero - 1 + find(W.New_Waveform(S.Zero:end) < S.Peak2Trough75,1);
    P.Parameter.Width75Peak2Trough = (P.Peak2Trough2P - P.Peak2Trough1P)/Resample_SR;
    NeuronsParams{cont,5} = P.Parameter.Width75Peak2Trough;
    
    
    P.BaselineCross1P= S.FirstPeak - 1 + find(W.New_Waveform(S.FirstPeak:S.Zero) < S.BaselineAmplitude,1);
    P.BaselineCross2P = S.Zero - 1 + find(W.New_Waveform(S.Zero:end) < S.BaselineAmplitude,1);
    P.Parameter.WidthBaselineCross = (P.BaselineCross2P - P.BaselineCross1P)/Resample_SR;
    NeuronsParams{cont,6} = P.Parameter.WidthBaselineCross;
    
    P.Baseline2Trough1P = find(W.New_Waveform(1:S.Zero) < S.Baseline2Trough75,1);
    P.Baseline2Trough2P = S.Zero - 1 + find(W.New_Waveform(S.Zero:end) < S.Baseline2Trough75,1);
    P.Parameter.Width75Baseline2Trough = (P.Baseline2Trough2P - P.Baseline2Trough1P)/Resample_SR;
    NeuronsParams{cont,7} =  P.Parameter.Width75Baseline2Trough;
    
    ParamStruct{cont} = P.Parameter;
    msg_clas = "Extracting Spikes: "+num2str(cont/length(Clus)*100)+"%";
    waitbar(cont/length(Clus),f,msg_clas)
    cont = cont + 1;
    %clearvars W S P
end
close(f)
Spike_Parameters = [cell2mat(NeuronsParams) Clus'];