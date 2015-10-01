right_rat_hmag = zeros(1440,1);
right_rat_rc = zeros(1440,2);
right_rat_fvec = zeros(1440,1);
for t = 600:24:840
    fprintf(1,'Working on %d ...\n',t);
    [hmag,rc,fvec] = compute_optical_flow_features(mov(t:t+24),0);
    right_rat_hmag(t:t+24) = hmag;
    right_rat_rc(t:t+24,:) = rc;
    right_rat_fvec(t:t+24) = fvec;
end
