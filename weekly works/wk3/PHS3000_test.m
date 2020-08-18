% This is a test file for seeing how scripts behave.
load pendulum_data.mat %reads file and imports to workspace

plot(y) % plot the values of y against index
y_bar=mean(y); % calculates mean of y
y_std=std(y); % calculates std dev (N-1) of y

