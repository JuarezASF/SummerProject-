function display_images(mov,low,high,cols)
   nFrames = high - low;
   rows = ceil(nFrames/cols);
   figure; 
 
   for i = 1:rows
       for j = 1:cols
           subplot(rows,cols,(i-1)*cols + j); 
           
           imagesc(mov(low-1 + (i-1)*cols + j).cdata);
       end
   end
end
           