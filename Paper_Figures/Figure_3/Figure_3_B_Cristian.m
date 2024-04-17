clear;
close all;
clc;

data_path = 'D:/_work_cestarellas/Analysis/Pack_Daniel_project/Figures/3_Figure/Data_Figure_3/';


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

% Trial section: Response
CP_rec_Res =[];
CP_gen_Res =[];
files = dir([data_path 'Resp*']);
res_files = {files.name};
for i=1:length(tot_files)
    load([data_path res_files{i}])
    model_gen=parcs(Gen,M_change_point);
    model_rec=parcs(Rec,M_change_point);
    CP_rec_Res = [CP_rec_Res model_rec.ch'];
    CP_gen_Res = [CP_gen_Res model_gen.ch'];
end

% Trial section: Reward
CP_rec_Rew =[];
CP_gen_Rew =[];
files = dir([data_path 'Rew*']);
rew_files = {files.name};
for i=1:length(tot_files)
    load([data_path rew_files{i}])
    model_gen=parcs(Gen,M_change_point);
    model_rec=parcs(Rec,M_change_point);
    CP_rec_Rew = [CP_rec_Rew model_rec.ch'];
    CP_gen_Rew = [CP_gen_Rew model_gen.ch'];
end








subplot(1,3,1)
plot(0.1:200, 0.1:200, 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
hold on;
scatter(CP_rec_Tot,CP_gen_Tot,25,"blue","filled")
xlabel("Neural change point rec.")
ylabel("Neural change point gen.")
title("Total")
subplot(1,3,2)
plot(0.1:200, 0.1:200, 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
hold on;
scatter(CP_rec_Res,CP_gen_Res,25,"blue","filled")
xlabel("Neural change point rec.")
ylabel("Neural change point gen.")
title("Execution")
subplot(1,3,3)
plot(0.1:200, 0.1:200, 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
hold on;
scatter(CP_rec_Rew,CP_gen_Rew,25,"blue","filled")
xlabel("Neural change point rec.")
ylabel("Neural change point gen.")
title("Reward")




























