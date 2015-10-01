% code for detecting clonus
% you  max magnitude of optical flow 
%      angle of the max magnitude
%      location of max magnitude
% Algorithmically, a clonus is a section where the peaks of the angle
% and the magnitude coincide and the max is at least one std over the mean
% of the entire segment

[pks,locs] = findpeaks(left_rat_hmag(600:840),'MinPeakHeight',mean(left_rat_hmag(600:840)),...
    'MinPeakDistance',4);
[pks1,locs1] = findpeaks(left_rat_fvec(600:840),'MinPeakHeight',mean(left_rat_fvec(600:840)),...
    'MinPeakDistance',4);
