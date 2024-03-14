function root_mean_square_error = RMSE(state_sequence1,state_sequence2)
%RMSE calculates the root mean square error between state_sequence1 and
%state_sequence2 
%INPUT: state_sequence1: a cell array of size (total tracking time x 1),
%       each cell contains an object state mean vector of size
%       (state dimension x 1)
%       state_sequence2: a cell array of size (total tracking time x 1),
%       each cell contains an object state mean vector of size
%       (state dimension x 1)
%OUTPUT:root_mean_square_error: root mean square error --- scalar

num_samples = length(state_sequence1)  ;
err = cell2mat(state_sequence1) - cell2mat(state_sequence2) ;
root_mean_square_error = sqrt(mean(err.^2)) ;
end