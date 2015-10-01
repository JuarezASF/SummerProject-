% process the videos 24 frames at a time, writing temporary results in
% hmag and rc

left_rat_hmag = zeros(1440,1);
left_rat_rc = zeros(1440,2);
left_rat_fvec = zeros(1440,1);
for t = 600:24:840
    fprintf(1,'Working on %d ...\n',t);
    [hmag,rc,fvec] = compute_optical_flow_features(mov(t:t+24),1);
    left_rat_hmag(t:t+24) = hmag;
    left_rat_rc(t:t+24,:) = rc;
    left_rat_fvec(t:t+24) = fvec;
end

