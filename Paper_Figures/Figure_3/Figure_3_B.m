clear;
close all;
clc;
restoredefaultpath()
addpath('/zi-flstorage/Max.Thurm/PhD/Paper1_plots/pub_change_points/')
addpath('/zi-flstorage/Max.Thurm/PhD/PLRNN_in_use/')
addpath('/zi-flstorage/Max.Thurm/PhD/PLRNN_in_use/regPLRNN/')


data_path = '/zi-flstorage/Max.Thurm/PhD/Paper1_plots/best_fits/mat_best_anlv_3/';
files = dir(data_path);
files = {files.name};
files = files(3:end);

res_name = 'EM_structure';

event = 1; %cue light: 1; lever presentation: 2; reward: 3; Baseline: 4; Total: 5
titles = {'Cue light', 'Lever presentation', 'Reward', 'Baseline', 'Total'};





fig_save_path = '/zi-flstorage/Max.Thurm/PhD/Paper1_plots/pub_change_points/';

for xtr = 1:5
    CP_REC = nan(1, length(files));
    CP_GEN = nan(1, length(files));
    event = xtr;
    for i = 1:length(files)
        file = files{i};
        res_name = file(1:end-4);
        load([data_path file])


        eval(['collection = ', res_name, '.collection;']);
        eval(['net = ', res_name, '.net3;']); %!!!
        eval(['X = ', res_name, '.X;']);
        eval(['Inp_ = ', res_name, '.Inp;']);
        eval(['Inp = ', res_name , '.Inp;']);
        eval(['Ezi = ', res_name , '.Ezi3;']); %!!!
        eval(['vr = collection.VR_trials;']);
        eval(['sr = collection.SR_trials;']);
        cp = max([sr vr]);
        ntr = length(X);
        cue_side = collection.y;
        response = collection.z;
        reward = collection.u;
        q = size(X{1}, 1);
        p = net.p;

        rule = zeros(1, ntr);
        rule(cp:end) = 1;

        [Z_gen, X_gen] = run_networ_exp(net, Inp_, 0);

        try collection.excluded_trials
            mask = collection.excluded_trials;

            response_time = collection.x(mask);
            cue_side = collection.y(mask);
            response = collection.z(mask);
            reward = collection.u(mask);

            rule = zeros(1, ntr);
            rule(cp:end) = 1;
            rule = rule(mask);
        catch 

        end

        analysis_data = {'X', 'X_gen', 'Ezi', 'Z_gen'};
        window = [0 20]; 
        data = analysis_data{1}; %1: X; 2: X_gen; 3: Ezi; 4: Z_gen
        if strcmp(data, 'X') || strcmp(data, 'X_gen')
            n_states = q;
        else
            n_states = p;
        end

        if event == 4
            cts = ones(1, ntr);
            window = [0 40];
        elseif event == 5
            cts = ones(1, ntr);
            window = [0 0];
        else
            action_times = get_times(Inp);
            cts = action_times(event,:);
        end

        test_aves = calc_window_aves(X, cts, window);
        test_aves_gen = calc_window_aves(X_gen, cts, window);

        model_rec = parcs(test_aves.', 1);
        model_gen = parcs(test_aves_gen.', 1);

        cp_rec = model_rec.ch;
        cp_gen = model_gen.ch;

        CP_REC = [CP_REC cp_rec];
        CP_GEN = [CP_GEN cp_gen];

        disp(i);
    end

    fig = figure();
    plot(0.1:200, 0.1:200, 'Color', [0.5 0.5 0.5], 'LineWidth', 2);
    hold on;
%     scatter(CP_REC, CP_GEN, 80, 'filled');
    scatter(CP_REC, CP_GEN, 80, [0 0 0], 'filled');
    xlim([0 200])
    ylim([0 200])
    xlabel('recorded change points')
    ylabel('simulated change points')
%     title(titles{event})
    ax = gca;
    ax.FontSize = 20;
    ax.FontName = 'Arial';
%     set(fig, 'color', 'none')
    saveas(fig, [fig_save_path titles{event} '.png']);
%     exportgraphics(fig, [fig_save_path titles{event} '.svg'],...
%         'Resolution', 600, 'ContentType', 'vector', 'BackgroundColor', 'none')

end
