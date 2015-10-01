function make_clip(mov,low,high)
    vobj = VideoWriter('clip.mp4');
    open(vobj);
    for i = low:high
       writeVideo(vobj,mov(i).cdata);
    end
    close(vobj);