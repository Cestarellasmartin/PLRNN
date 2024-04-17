%Developer: Cristian Estarellas Martin
%Date: 27_Oct_2023
%-----------------IT IS VALID FOR GAMBLING TASK-2022-----------------------
% Script to pre-process the experimental data
% This program will organise the raw data in different variables:
% Spike Activity:Sampling points when the neuron generates an action potential 
% Spike Parameters: Waveform properties of each neuron
% Wheel Movement: Degrees of the wheel during the task
% Metadata: general information about the experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The script will ask you some inputs in different formats: int (integer)
% or str (string). That information will be saved in the Metadata structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% Select the directory where are the data to preprocess
Path = uigetdir;
cd(Path)

% Needed Inputs
Amp = dir('amplifier.dat');
Tim = dir('time.dat');

Nch= round(4*Amp.bytes/(2*Tim.bytes));                                      %Check that the rounding works;
disp(['Number of Channels: ',num2str(Nch)])

% Information about the Experiment
prompt_type='Animal Type (Example (str): Mice C56/BL6):' ;
prompt_code='Animal code (Example (str): CEM01): ';
prompt_experimenter='Name Experimenter (str): ';
prompt_age='Age animal (Example (str): 8 months): ';
prompt_recording='Recording method (Example (str):Acute silicon Probe.128 channels): ';
prompt_areas='Brain areas (Example (str): Prelimbic Cortex): ';
% Name of files
% Each file generated will start with this name
prompt_save_name='Name to save the file (str) - JG15_mmddyy): ';
filename=input(prompt_save_name);

% Used in Metadata Block
Metadata.type=input(prompt_type);                                           %"Mice C56/BL6";
Metadata.animalcode=input(prompt_code);                                     %"CEM01";
Metadata.experimenter=input(prompt_experimenter);                           %"Cristian";
Metadata.age=input(prompt_age);                                             %"8 months";
Metadata.recordingmethod= input(prompt_recording);                          %"Acute silicont Probe. 128 channels";
Metadata.brainareas=input(prompt_areas) ;                                   %"Orbito-Frontal Cortex";

%% Loading data
disp('Loading files...')

% Loading Raw Data
raw_fid = fopen('amplifier.dat');
raw_data = fread(raw_fid,[Nch,inf],'int16');                                % reading the file. Int16 is normally in intan. Axone works with int32
fclose(raw_fid);
% Loading Digital Signal
fileinfo = dir('digitalin.dat');
num_samples = fileinfo.bytes/2;                                             % uint16 = 2 bytes
fid = fopen('digitalin.dat', 'r');
digital_word = fread(fid, num_samples, 'uint16');
fclose(fid);

disp('Files loaded')

%% SAMPLING RATE
SR = 20000;                                                                 % Sampling Rate
max_time=size(raw_data,2)/SR;                                               % Time in seconds
warning(['Default Sampling Rate (Hz):',num2str(SR)])
disp(['Total Time of Recordings (s):',num2str(max_time)])
warning('Change the Sampling Rate in the script if it is necessary (line:70)')

%% Information of the experiment (variable Metadata)
disp('Generating Metadata variable.')

% Metadata: 
% Ideally, a structure variable that provides sufficient information for your experiment 

Metadata.laboratory='Cognitive Neurobiology-TK lab';                        % laboratory
Metadata.date=fileinfo.date;                                                % date
time_start=duration(fileinfo.date(end-7:end));                              
time_end=time_start+seconds(max_time);                                      
Metadata.time=char([time_start,time_end]);                                  % initial & final time
Metadata.SR = SR;                                                           % Sampling Rate
Metadata.experimentaltask='Gambling_Head_Fixed';                            % Experimental Task
Metadata.spikesortinginfo='Kilosort.';                                      % Spike Sorting Info
Metadata.Behaviour_Info=Behavioural_function;                               % Table with the column info of the Trial_Sync data

disp('Metadata Generated')

%% Spike data (variable: STMtx)
% 1) STMtx (= spike timing matrix), with colums = units, rows = spike times 
% in SAMPLING UNITS
% ▪	Note: 
% •	Units may have a different number of spikes such that units
% with less spikes have additional NaNs filling up the bottom part of the
% matrix 
% •	STMtx should ONLY contain spike times, no header lines for unit
% numbers 

disp('Generating STMtx variable')

sp = Extracting_Cluster_Data(Path);                                         % Function from Kilosort package
Clus=sp.cids(sp.cgs==2);                                                    % Selection of good clusters ("neurons")

length_sp=sum(sp.clu == Clus);
STMtx=nan(max(length_sp),length(Clus));
unit=0;
for Neuron = Clus
    unit=unit+1;
    STMtx(1:length_sp(unit),unit)=sp.st(sp.clu == Neuron)*SR;
end

%% Spike Parameters

disp('Extracting waveform features...')

[~, ~, ~, Spike_Parameters] = SpikeParameters;                              % Hugo's Function

unknow_cell = find(Spike_Parameters(:,1) ==0);
Spike_Parameters(unknow_cell,1:7) = nan;
var_names = ["FR","Peak2Peak","Trough2LastPeak","Firstpeak2Trough",...
    "Width75Peak2Trough","WidthBaselineCross","Width75Baseline2Trough","Clusters"];
% Generation of Table
Spike_features = array2table(Spike_Parameters);
Spike_features.Properties.VariableNames = var_names;

disp('Extraction finished')

%% Wheel Movement

disp('Computing Wheel Movement:')

Wheel = wheel_mov_function(digital_word);

%% LFP filter (Optional)
% Change LTP_Trigger to true if you want to compute the LFP of the rawdata
LFP_Trigger = false;

if (LFP_Trigger==true)
    channels = [1,4,5,8,9,6,41,100];                                        % array with the channels to compute the LFP filtered by a lowpass of 300Hz
    LFP = LFP_preprocess(SR,raw_data,max_time,channels);
    % Save LFP data
    save_name = strcat(filename,'_LFP_LowPass.mat');
    save(save_name,"LFP",'-v7.3')
end

%% Save Data

% Spike Activity
save_name = strcat(filename,'_SpikeActivity.mat');
save(save_name,"STMtx")

% Metadata
save_name = strcat(filename,'_Metadata.mat');
save(save_name,"Metadata")

% Spike Parameters
save_name = strcat(filename,'_SpikeParameters.mat');
save(save_name,"Spike_features")

% Wheel Movement
save_name = strcat(filename,'_WheelMovement.mat');
save(save_name,"Wheel")

disp("Data Saved!")

clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% LOCAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Beh_info = Behavioural_function
    Names_TrialSync{1,1}='Tria_Start_s';                                        % (seconds from behavioural start)';
    Names_TrialSync{2,1}='Trial_End_s';                                         % (seconds from behavioural start)';
    Names_TrialSync{3,1}='Trial_duration_s';                                    % (from the synchronisation)';
    Names_TrialSync{4,1}='Block_number';
    Names_TrialSync{5,1}='Gamble_Arm';                                          % (Right=1 Left=0)';
    Names_TrialSync{6,1}='Prob_Big_Reward';
    Names_TrialSync{7,1}='Prob_Small_Reward';
    Names_TrialSync{8,1}='Ammount_Big_Reward';
    Names_TrialSync{9,1}='Ammount_Small_Reward';
    Names_TrialSync{10,1}='Wheelt_not_stopping';
    Names_TrialSync{11,1}='Not_responding_trial';
    Names_TrialSync{12,1}='Chosen_side';                                        % (Right=1 Left=0)';
    Names_TrialSync{13,1}='Chosen_arm';                                         % (Gamble=1 Safe=0)';
    Names_TrialSync{14,1}='Rewarded_trial';
    Names_TrialSync{15,1}='Start_trial_smplpts';
    Names_TrialSync{16,1}='Cue_presentation_smplpts';
    Names_TrialSync{17,1}='Response_smplpts';
    Names_TrialSync{18,1}='Reward_period_smplpts';
    Names_TrialSync{19,1}='End_trial_smplpts';
    Columns = transpose(1:size(Names_TrialSync,1));
    Beh_info=table(Columns,Names_TrialSync);
end