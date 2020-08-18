function data_mod=PHS_offset(input_data)

% this is a function to centre data about zero and normalise to a maximum 
% of 1. Assumes data is a vector (not a 2D Matrix).
% Usage: output=PHS3000_offset(input)

% "input_data" is the name of the variable in the function assigned to the
% input argument.
% the variablein the script below named "data_mod" is exported by the
% function as an output

data=input_data-mean(input_data); % Here we substract off the mean value of
% the data
max_data=max(data); % find maximum of the shited data
data_mod=data/max_data; % nomalise
end
