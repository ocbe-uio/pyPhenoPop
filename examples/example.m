% Turn off figures popping up 
%set(0,'DefaultFigureVisible','off')

NR = 5;
Conc = [0.1, 1, 10, 50, 100, 500, 1000, 10000];
Time = [0,24,48,72,96];
filename = "./data/CLL_data_blinded/cleanfiles/flat_T0_data.csv";
%perform_inference(experiment_name, NR, Concentration_array, Time_array, filename, max_no_populations_optional, num_optim_optional, lower_bounds_optional, upper_bounds_optional, USE_TWO_NOISE_LEVELS_optional, ConcT_optional, TimeT_optional) 
perform_inference("CLL_data", NR, Conc, Time, filename, 2, 10); % , [-0.1, 0, 1e-6,   0, 1e-6], [ 0.1, 1,  2e3, 100, 5e4]); %, USE_TWO_NOISE_LEVELS_optional, ConcT_optional, TimeT_optional)

