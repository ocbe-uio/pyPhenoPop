% Turn off figures popping up 
set(0,'DefaultFigureVisible','off')
tStartOuter = tic;

% Compute negative loglikelihood for real data, fitting 1,2,3,+++ populations
% 06.10.2022
NR = 5;
NC = 8;
NT = 5;
Conc = [0.1, 1, 10, 50, 100, 500, 1000, 10000];
Time = [0,24,48,72,96];
% Load csv into Matlab 
flat_T0_data = table2array(readtable("./data/CLL_data_blinded/cleanfiles/flat_T0_data.csv"));
flat_T1_data = table2array(readtable("./data/CLL_data_blinded/cleanfiles/flat_T1_data.csv"));
flat_A_data = table2array(readtable("./data/CLL_data_blinded/cleanfiles/flat_A_data.csv"));
flat_B_data = table2array(readtable("./data/CLL_data_blinded/cleanfiles/flat_B_data.csv"));
flat_C_data = table2array(readtable("./data/CLL_data_blinded/cleanfiles/flat_C_data.csv"));
% Reshape
%x_finals_temp = zeros(1, max_no_populations, 5*max_no_populations);
%x_finals_temp(1, ii, 1:length(ub)) = x_final';
T0_data = zeros(NR,NC,NT);
T1_data = zeros(NR,NC,NT);
A_data = zeros(NR,NC,NT);
B_data = zeros(NR,NC,NT);
C_data = zeros(NR,NC,NT);
for time_index = 0:NT-1 
    for conc_index = 1:NC
        T0_data(1:NR,conc_index,time_index+1) = squeeze(flat_T0_data(NR*time_index+1:NR*(time_index+1),conc_index));
        T1_data(1:NR,conc_index,time_index+1) = squeeze(flat_T1_data(NR*time_index+1:NR*(time_index+1),conc_index));
        A_data(1:NR,conc_index,time_index+1) = squeeze(flat_A_data(NR*time_index+1:NR*(time_index+1),conc_index));
        B_data(1:NR,conc_index,time_index+1) = squeeze(flat_B_data(NR*time_index+1:NR*(time_index+1),conc_index));
        C_data(1:NR,conc_index,time_index+1) = squeeze(flat_C_data(NR*time_index+1:NR*(time_index+1),conc_index));
    end 
end 

% Remove time 0
Time = Time(1:NT-1); % 24 is the new zero
T0_data = T0_data(1:NR,1:NC,2:NT);
T1_data = T1_data(1:NR,1:NC,2:NT);
A_data = A_data(1:NR,1:NC,2:NT);
B_data = B_data(1:NR,1:NC,2:NT);
C_data = C_data(1:NR,1:NC,2:NT);
NT = NT-1;

case_array = ["B", "C"]; %["T0", "T1", "A", "B", "C"];
for patient_index = 1:length(case_array)
tStart = tic;
case_name = case_array(patient_index)

num_optim = 3000 % Number of starting points in maximum likelihood optimization.
%highest_rate = 0.1 % Not used when using optimization_constraints_real_data   % Highest exponential rate we assume there may be
max_no_populations = 3 % The highest number of populations to try fitting to the data.
seednr_1 = 45 % Seed for the optimization initialization
lower_limit_mixparam = 0 % Each inferred mixture parameter must be greater than this.
lower_limit_E_ratio = 1 % The smallest ratio (E_i / E_j) among inferred E parameters must be greater than this.
USE_E_THRESHOLD = false
INFER_SIGMA = true % If true, sigma is estimated by MLE. If false, the true value is given.
USE_TWO_NOISE_LEVELS = true
if USE_TWO_NOISE_LEVELS
    ConcT = 10;
    TimeT = 48;
    if ~INFER_SIGMA
        error("If using two noise level, sigma must also be inferred or you must implement an estimate for sigma")
    end
end
PLOT_DATA = true
FIT_THINGS = true
PLOT_FINAL_FIT = true

%x_finals = zeros(N_settings, 1, max_no_populations, 5*max_no_populations); % Estimated parameters on training sets (k-1 folds)
%f_vals = Inf(N_settings, 1, max_no_populations); % negative loglikelihood values on training sets (k-1 folds)
%best_no_pops_from_average_loglikelihood = zeros(1, N_settings); % The number of populations with highest average loglikelihood score across k folds, for N_settings
%best_no_pops_from_bic = zeros(1, N_settings);
%average_loglikelihoods = zeros(N_settings, max_no_populations);
%bic_values = zeros(N_settings, max_no_populations);

colors_estimated = [        
    0.9570 0.6640 0.2578
    0.105468750000000   0.617187500000000   0.464843750000000
    0.9570 0.2578 0.5039
    0.2578 0.9570 0.6172 
    0.7578 0.2578 0.9570 
    0                   0   0.726562000000000
    0.957000000000000   0.257800000000000   0.503900000000000
];

switch case_name
case "T0"
    DATA = T0_data;
case "T1"
    DATA = T1_data;
case "A"
    DATA = A_data;
case "B"
    DATA = B_data;
case "C"
    DATA = C_data;
end

R = size(DATA, 1)
N_c = size(DATA, 2)
N_t = size(DATA, 3)

x_finals_temp = zeros(1, max_no_populations, 5*max_no_populations);
f_vals_temp = Inf(1, max_no_populations);
negative_loglikelihood_values = Inf(1,max_no_populations);

conclabels = strings(1,N_c);
for c_index=1:N_c
    conclabels(c_index) = num2str(Conc(c_index));
end
newcolors = [0.267004 0.004874 0.329415
0.282623 0.140926 0.457517
0.253935 0.265254 0.529983
0.206756 0.371758 0.553117
0.163625 0.471133 0.558148
0.127568 0.566949 0.550556
0.134692 0.658636 0.517649
0.266941 0.748751 0.440573
0.477504 0.821444 0.318195
0.741388 0.873449 0.149561
0.993248 0.906157 0.143936];
newcolors = [
    repmat([0.267004 0.004874 0.329415],R,1)
    %repmat([0.282623 0.140926 0.457517],R,1)
    repmat([0.253935 0.265254 0.529983],R,1)
    repmat([0.206756 0.371758 0.553117],R,1)
    repmat([0.163625 0.471133 0.558148],R,1)
    repmat([0.127568 0.566949 0.550556],R,1)
    repmat([0.134692 0.658636 0.517649],R,1)
    repmat([0.266941 0.748751 0.440573],R,1)
    repmat([0.477504 0.821444 0.318195],R,1)
    %repmat([0.741388 0.873449 0.149561],R,1)
    repmat([0.993248 0.906157 0.143936],R,1)
    ]; 

if PLOT_DATA
    repeated_colors = [
        repmat([0.9570 0.2578 0.5039],R,1)
        repmat([0.2578 0.5273 0.9570],R,1)
        repmat([0.9570 0.6640 0.2578],R,1)
        repmat([0.2578 0.9570 0.6172 ],R,1)
        repmat([0.7578 0.2578 0.9570 ],R,1)
        repmat([0                   0   0.726562000000000],R,1)
        repmat([0.957000000000000   0.257800000000000   0.503900000000000],R,1)
        repmat([0.257800000000000   0.527300000000000   0.957000000000000],R,1)
        repmat([0.957000000000000   0.664000000000000   0.257800000000000],R,1)
    ]; 
    
    % Plot observationss
    fig = figure; %('Position',[800 800 400 300]);
    movegui(fig,[1275 630]); % x y positions of bottom left corner
    h = axes;
    set(h,'xscale','log')
    colororder(repeated_colors);
    min_x = 0.1*min(Conc(2:N_c));
    max_x = 10*max(Conc(2:N_c));
    
    % Set time and conc
    increment = 1; % Step in time. Increase to plot less of the data points
    plot_Time = Time(1:increment:N_t); %[0 24 48 72 96]; %[0 12 24 26 48 60 72 84 96];
    plot_N_t = length(plot_Time);
    tlabels = strings(1,N_t);
    for t_index=1:plot_N_t
        tlabels(plot_N_t-t_index+1) = strcat("T=", int2str(plot_Time(t_index)));
    end
    plot_Conc = [min_x Conc(2:N_c)];
    for t_index=plot_N_t:-1:1
        hold on
        for r_index = 1:R
            if r_index == 1
                % Show handle
                semilogx(plot_Conc, DATA(r_index,1:N_c,increment*t_index - (increment-1)), '.', 'MarkerSize', 10,'HandleVisibility','on')
            else
                %semilogx(Conc(2:N_c), DATA(r_index,2:N_c,increment*t_index - (increment-1)), '.', 'MarkerSize', 10,'HandleVisibility','off')
                % Include Conc 0 data
                semilogx(plot_Conc, DATA(r_index,1:N_c,increment*t_index - (increment-1)), '.', 'MarkerSize', 10,'HandleVisibility','off')
                %semilogx(min_x, DATA(r_index,1,increment*t_index - (increment-1)), '.', 'MarkerSize', 10,'HandleVisibility','off')
            end
        end
    end
    for c_index=2:N_c
        xline(Conc(c_index), "k",'HandleVisibility','off')
    end
    ylabel('Cell count')
    xlabel('Drug dose')
    xlim([min_x, max_x])
    legend(tlabels)
    title([strcat(newline, case_name) 'Cell counts' '1 dot per replicate, dose 0 included'])
    saveas(gcf, [pwd, '/data/CLL_data_blinded/plots/logl-data-', num2str(case_name), '.png'])

    % Plot in time 
    figure
    colororder(newcolors);
    DATA(r_index,c_index,1:N_t)
    hold on
    for c_index=1:N_c
        for r_index=1:R
            plot(Time, squeeze(DATA(r_index,c_index,1:N_t)), '.-', 'MarkerSize', 10,'HandleVisibility','off')
        end
    end
    ylabel('Cell count')
    xlabel('Time (h)')
    ylim([0 inf])
    %legend(conclabels(1:N_c), 'location', 'northeast') % did not work, empty
    title([strcat(newline, case_name) 'All replicates for each concentration'])
    saveas(gcf, [pwd, '/data/CLL_data_blinded/plots/logl-data-every-replicate-', num2str(case_name), '.png'])
end % if PLOT_DATA

if FIT_THINGS
% Fit with 1,2,3,+++ subpopulations
options = optimoptions(@fmincon,'FiniteDifferenceType','central','SpecifyObjectiveGradient',false,'Display','off','MaxFunctionEvaluations',5990,'MaxIterations',5990,'OptimalityTolerance',1.0e-10,'StepTolerance',1.0000e-10);
mydate = sprintf('%s', datestr(now,'yyyy-mm-dd'));

for ii = 1:max_no_populations
    tStart_inner = tic;
    % Infer ii populations
    no_populations = ii

    % Set upper and lower bounds for [alpha, b, E, n, sigma(both Hi, Lo)]
    % The E limits are then converted to log values for log spaced sampling
    switch case_name
    case "T0"
        lb_all = [-0.1, 0, 1e-6,   0, 1e-6];
        ub_all = [ 0.1, 1,  2e3, 100, 5e4];
    case "T1"
        lb_all = [-0.1, 0, 1e-6,   0, 1e-6];
        ub_all = [ 0.1, 1,  5e4, 100, 1e6];
    case "A"
        lb_all = [-0.1, 0, 1e-6,   0, 1e-6];
        ub_all = [ 0.1, 1,  5e4, 100, 1e6];
    case "B"
        lb_all = [-0.1, 0, 1e-6,   0, 1e-6];
        ub_all = [ 0.1, 1,  5e4, 100, 1e6];
    case "C"
        lb_all = [-0.1, 0, 1e-6,   0, 1e-6];
        ub_all = [ 0.1, 1,  5e4, 100, 1e6];
    end
    lb_all(3) = log10(lb_all(3));
    ub_all(3) = log10(ub_all(3));

    [lb, ub, A_inequality, b_inequality, nonlcon] = optimization_constraints_real_data(no_populations, INFER_SIGMA, USE_TWO_NOISE_LEVELS, lower_limit_mixparam, lower_limit_E_ratio, lb_all, ub_all);
    
    if INFER_SIGMA
        if USE_TWO_NOISE_LEVELS
            f=@(x)vectorized_objective_function_k_subpop_and_two_noise_levels(x,no_populations,DATA,Conc,Time,R,ConcT,TimeT);
        else
            f=@(x)vectorized_objective_function_k_subpopulations(x,no_populations,DATA,Conc,Time,R);
        end
    else
        if USE_TWO_NOISE_LEVELS
            error("Again: If using two noise level, sigma must also be inferred or you must implement an estimate for sigma")
        else 
            % Estimate the standard deviation 
            std_estimate = std(DATA); % estimates the standard deviation for all (time, concentration) combinations
            Noise_estimate = mean(std_estimate, 'all'); % Finds the average noise estimate instead of using particulars
            f=@(x)vectorized_objective_function_k_subpopulations([x' Noise_estimate]',no_populations,DATA,Conc,Time,R); % Drop Noise parameter
            %f=@(x)vectorized_objective_function_k_subpopulations([x' Noise]',no_populations,DATA,Conc,Time,R); % Drop Noise parameter
        end
    end
    fval=inf;
    
    rng(42); %set seed to start iterations the same place for every version of R
    for nn=1:num_optim
        x0=rand(length(ub),1).*(ub-lb) + lb;
        x0(3:5:5*no_populations-2)=10.^(x0(3:5:5*no_populations-2));
        lb(3:5:5*no_populations-2)=1e-6; %To avoid Inf or NaN in log() terms we require positive E values 
        ub(3:5:5*no_populations-2)=1e4; % E
        if USE_E_THRESHOLD
            [xx,ff]=fmincon(f,x0,A_inequality,b_inequality,[],[],lb,ub,nonlcon,options);
        else
            [xx,ff]=fmincon(f,x0,A_inequality,b_inequality,[],[],lb,ub,[],options);
        end
        if ff<fval
            x_final=xx;
            fval=ff;
        end    
    end
    x_finals_temp(1, ii, 1:length(ub)) = x_final';
    f_vals_temp(1, ii) = fval;
    negative_loglikelihood_values(ii) = fval
    % BIC
    N_observations = N_c*N_t*R;
    Nparams = length(ub);
    % BIC:         %         k          *        ln(N)        + 2 * negative loglikelihood
    %bic_values_temp(ii) = Nparams * log(N_observations) + 2 * fval;

    inferred_mixtures = squeeze(x_finals_temp(1, no_populations, 5:5:5*(no_populations-1)))';
    inferred_mixtures = [inferred_mixtures, 1-sum(inferred_mixtures)];
    sorted_inferred_parameters = squeeze(x_finals_temp(1, no_populations,1:5*no_populations))';
    inferred_GR50s = zeros(1, no_populations);

    if PLOT_FINAL_FIT
        plot_N_c = 1000;
        x = zeros(1,plot_N_c);
        x(2:plot_N_c) = logspace(log10(min_x),log10(max_x),(plot_N_c-1));    
        % Plot all the inferred cell lines 
        newcolors0 = colors_estimated(1:no_populations,:);
        fig = figure;
        colororder(newcolors0);
        %movegui(fig,[1275 100]); % x y positions of bottom left corner
        h = axes;
        set(h,'xscale','log')
        hold on
        for jj=1:no_populations
            inferred_parameters = sorted_inferred_parameters(5*jj-4:5*jj-1);
            y_inferred = ratefunc(inferred_parameters', x);
            semilogx(x,y_inferred, '--', 'LineWidth', 3)
            inferred_GR50s(jj) = find_gr50(inferred_parameters, Conc);
        end
        ylabel('Growth rate')
        xlabel('Drug concentration')  
        title([strcat(newline, case_name) strcat("Inferred growth rates")])
        inferred_legends = {};
        for jj = 1:no_populations
            str = strjoin({'Clone ' num2str(jj) ' Mixture ' num2str(inferred_mixtures(jj), '%.2f') ' GR50 ' num2str(inferred_GR50s(jj), '%.2e') });
            inferred_legends = [inferred_legends str];
        end
        % These concentration placements are not informative; Look at data plotting
        %for c_index=2:N_c
        %    xline(Conc(c_index), "k",'HandleVisibility','off')
        %end
        legend([inferred_legends],'Location','southwest')
        saveas(gcf, [pwd, '/data/CLL_data_blinded/plots/logl-inferred_populations-', num2str(case_name), '-no-pop-', num2str(no_populations), '-num_optim-', num2str(num_optim), '.png'])
    end % if PLOT_FINAL_FIT
    tEnd_inner = toc(tStart_inner);
end % of inference for ii populations loop

negative_loglikelihood_values
inferred_GR50s
" Parameters 1 pop"
squeeze(x_finals_temp(1,1,1:6))
" Parameters 2 pops"
squeeze(x_finals_temp(1,2,1:11))

savestr = strcat(num2str(case_name), '-num_optim-', num2str(num_optim));

% Save data
save([strcat('./data/CLL_data_blinded/negLL/negative_loglikelihood_and_GR50-', savestr, '.mat')], 'negative_loglikelihood_values', 'x_finals_temp', 'inferred_GR50s')

% Plot the negative loglikelihood values
fig = figure;
hold on
xlabel('Number of inferred populations')
ylabel('Negative loglikelihood')
title(case_name)
plot(1:max_no_populations, negative_loglikelihood_values, '.-k')
saveas(gcf, [pwd, '/data/CLL_data_blinded/plots/negative_loglikelihood_values-', savestr, '.png'])
end %%%% FIT_THINGS block
tEnd = toc(tStart)
end %%%% patient number loop
tEndOuter = toc(tStartOuter)
