% blob detector using matlab's conneccted components
% assume read_videos has been run before
figure;
start_index = 1;
for i = 1:800
    
    %i = 559;

    % Step 1: get one image

    im = mov(i).cdata;

    % Step 2: reduce its size
    im_half = imresize(im,0.5);


    % Focus on the right image(pick only the blue component)
    im_right = im_half(:,80:end,3);

    % threshold the "blue" image
    im_bw = im2bw(im_right,graythresh(im_right));

    % clean the thresholded image

    im_bw1 = bwmorph(im_bw,'open');
    im_bw2 = bwmorph(im_bw1,'fill');
    im_bw3 = bwmorph(im_bw2,'thicken');
    im_bw4 = bwmorph(im_bw3,'clean');
    new_im_bw = bwareaopen(im_bw4,50);



    % trace the boundaries
    [B,L,N] = bwboundaries(new_im_bw,'noholes');
    %figure; imshow(im_right); hold on;
    %figure; 
    imshow(label2rgb(L, @jet, [.5 .5 .5])); hold on;
    for k=1:length(B),
        boundary = B{k};
        if(k > N)
            plot(boundary(:,2),...
                boundary(:,1),'g','LineWidth',2);
        else
            plot(boundary(:,2),...
                boundary(:,1),'r','LineWidth',2);
        end
    end



    % compute the properties of the regions
    stats = regionprops(new_im_bw, im_right,'Area','Centroid','BoundingBox','MeanIntensity');
    centers = cat(1,stats.Centroid);
    scatter(centers(:,1),centers(:,2));
    areas = cat(1,stats.Area);
    intensities = cat(1,stats.MeanIntensity);
    bboxes = cat(1,stats.BoundingBox);

    % the rat is the region of about 120 to 400 pixels in size
    % with the maximum mean intensity
    potential_rat_indexes = find(areas >= 80 & areas <= 450);
    if length(potential_rat_indexes) == 1
        rat_index = potential_rat_indexes;
    else
        [maxin,mindex] = max(intensities(potential_rat_indexes));
        rat_index = potential_rat_indexes(mindex);
    end

    % check that the rat has not jumped too far between frames
    % it indicates an error in segmentation
    % then, find the blob that is closest to the old position

    pos = centers(rat_index,:);
    d = norm(pos-old_pos);
    if d < 5 || i == start_index
        % draw the bounding box for the rat
        rectangle('Position',[bboxes(rat_index,1),bboxes(rat_index,2),bboxes(rat_index,3),bboxes(rat_index,4)],'LineWidth',3);

        pause(0.01);
        old_pos = pos;
        fprintf(1,'Rat at position %.2f, %.2f distance %.2f at time %d \n',centers(rat_index,1), centers(rat_index,2),d,i);
    else
      fprintf(1,'Lost rat distance %.2f with (bad) location %.2f %.2f at time %d\n',d,centers(rat_index,1),centers(rat_index,2),i);
    end
end
