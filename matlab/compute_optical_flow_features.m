function [hmag,rc,fvec] = compute_optical_flow_features(mov,left_flag)

    % do an estimate of motion between two successive frames
     hmag = zeros(length(mov),1);
     rc = zeros(length(mov),2);
     fvec = zeros(length(mov),1);
     for k = 1:length(mov)-1
        fprintf(1,'k = %d \n',k);
        if left_flag == 1
           left_rat_k = mov(k).cdata(:,1:160,:);
           left_rat_k1 = mov(k+1).cdata(:,1:160,:);
           uv = estimate_flow_interface(left_rat_k,left_rat_k1,'classic+nl-fast');
        else
           right_rat_k = mov(k).cdata(:,161:320,:);
           right_rat_k1 = mov(k+1).cdata(:,161:320,:);
           uv = estimate_flow_interface(right_rat_k,right_rat_k1,'classic+nl-fast'); 
        end
        
        % find highest magnitude of optical flow in image and its row,col
        % location
        mag = uv(:,:,1).^2 + uv(:,:,2).^2;
        hmag(k) = max(max(mag));
        [r,c] = find(mag == hmag(k));
        
        
        
        % plot the location 
        % figure; imagesc(mag); colorbar; hold;
        % scatter(c,r,200,'MarkerEdgeColor','b','MarkerFaceColor','g','LineWidth',2.0);
  
        if length(r) > 1
            rc(k,1) = r(1);
        else rc(k,1) = r;
        end
        if length(c) > 1
            rc(k,2) = c(1);
        else rc(k,2) = c;
        end

        % find the angular orientation of the highest magnitude flow
        fvec(k) = atan(uv(rc(k,1),rc(k,2),2)/uv(rc(k,1),rc(k,2),1));
        

        % and plot optical flow
        %figure; subplot(1,2,1);imshow(uint8(flowToColor(uv)));  title('color coding');
        %subplot(1,2,2); plotflow(uv);   title('Vector plot');
end