%% read the videos in the 01/ directory
addpath ../flow_code/;
addpath ../flow_code/data ../flow_code/utils ../flow_code/utils/flowColorCode;
addpath ../flow_code/utils/downloaded;

clear alist dirIndex filelist

alist = dir('../video/mp4');
dirIndex = [alist.isdir];
filelist = {alist(~dirIndex).name}';

% read one video at a time
i = 1;
fname = strcat('../video/mp4/',char(filelist(i)));

vobj = VideoReader(fname);
fprintf(1,'Video %s has %d frames and has width %d and height %d\n',fname,...
    vobj.NumberOfFrames,vobj.Height,vobj.width);

% fps = 24, so to get 1 minute of video read 1440 frames at a time
% preallocate the movie object; 1 minute of video per chunk

mframesize = 1440;
num_mframes = ceil(vobj.NumberOfFrames/mframesize);

% which minute of the 10 minute video are we looking at?

which_minute = 1;

mov(1:mframesize) = ...
       struct('cdata',zeros(vobj.Height,vobj.Width,3,'uint8'));
   
% read one frame to mframesize frames

for k = 1:1:mframesize
        mov(k).cdata = read(vobj,k+(which_minute-1)*1440);
       
end

