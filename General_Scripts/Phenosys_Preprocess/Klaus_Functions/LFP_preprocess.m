function LFP = LFP_preprocess(SR,raw_data,max_time,channel_list)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
% SR: Sampling Rate
% raw_data: raw data of each channel
% max_time: time of the recording in seconds
% channel_list: array of channels to analyse 
%
% Outputs:
% LFP: structure with the following information
% •LFP.V: matrix with columns= electrodes, rows= voltage traces across time 
% •LFP.T: sampling times in sec (or, alternatively, sampling rate = 1 value) 
% •LFP.F: band pass filter settings if used 
% •LFP.C: Channels used 

disp('Generating LFP variable')

% Low pass filter
dec_order=4;
SR_dec=SR/dec_order;
progress=waitbar(0,'Filtering Channels');
disp('Decimating signal: Sampling rate from 20000 to 5000 (order 4)')
disp('Filtering...Lowpass 300 Hz')

% LFP.C
num_channels=length(channel_list);
LFP.C=channel_list;

%LFP.V
for itera=1:num_channels
    i_ch=channel_list(itera);
    data_dec=decimate(raw_data(i_ch,:),dec_order);
    [LFP.V(:,i_ch),filter]=lowpass(data_dec,300,SR_dec,'ImpulseResponse','iir','Steepness',0.95);
    msg_filter = "Filtering Channels..."+num2str(itera/num_channels*100)+"%";
    waitbar(itera/num_channels,progress,msg_filter)
end
close(progress)

disp('Channels filtered')

% LFP.T 
LFP.T=linspace(0,max_time,max_time*SR_dec)';

% LFP.F
LFP.F.Coefficients=filter.Coefficients;
LFP.F.DesignMethod=filter.DesignMethod;
LFP.F.FrequencyResponse=filter.FrequencyResponse;
LFP.F.ImpulseResponse=filter.ImpulseResponse;
LFP.F.PassbandFrequency=filter.PassbandFrequency;
LFP.F.PassbandRipple=filter.PassbandRipple;
LFP.F.SampleRate=filter.SampleRate;
LFP.F.StopbandAttenuation=filter.StopbandAttenuation;
LFP.F.StopbandFrequency=filter.StopbandFrequency;


