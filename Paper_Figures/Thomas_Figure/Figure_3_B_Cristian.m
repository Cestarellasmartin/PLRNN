clear;
close all;
clc;

data_path = 'D:/_work_cestarellas/Analysis/Pack_Daniel_project/Figures/Figure_change_point/';

M_change_point = 2;
% Total Trial
CP_rec_Tot =[];
CP_gen_Tot =[];
files = dir([data_path 'Tot*']);
tot_files = {files.name};
for i=1:length(tot_files)
    load([data_path tot_files{i}])
    model_gen=parcs(Gen,M_change_point);
    model_rec=parcs(Rec,M_change_point);
    CP_rec_Tot = [CP_rec_Tot model_rec.ch'];
    CP_gen_Tot = [CP_gen_Tot model_gen.ch'];
end

% Trial section: Beh
CP_Beh =[];
files = dir([data_path 'Beh*']);
res_files = {files.name};
for i=1:length(tot_files)
    load([data_path res_files{i}])
    model_beh=parcs(Beh',M_change_point);
    CP_Beh = [CP_Beh model_beh.ch'];
end

% Trial section: W parameters
CP_W1 =[];
CP_W2 =[];
files = dir([data_path 'Wparams*']);
rew_files = {files.name};
for i=1:length(tot_files)
    load([data_path rew_files{i}])
    model_w2=parcs(W2',M_change_point);
    model_w1=parcs(W1',M_change_point);
    CP_W1 = [CP_W1 model_w1.ch'];
    CP_W2 = [CP_W2 model_w2.ch'];
end



























