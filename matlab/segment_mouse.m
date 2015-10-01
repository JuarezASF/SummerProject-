% segment the mouse out in an image

% Step 1: get one image

im = mov(i).cdata;

% Step 2: reduce its size
im_half = imresize(im,0.5);


% Focus on the right image
im_right = im_half(:,80:end,3);

% cut out the 
thresh = multithresh(im_right,2);
seg_I = imquantize(im_right,thresh);
im_rgb = label2rgb(seg_I);
imshowpair(im_right,im_rgb,'montage')

BW = edge(im_right,'sobel');
[H,T,R] = hough(BW);
imshow(H,[],'XData',T,'YData',R,...
            'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
P  = houghpeaks(H,7,'threshold',ceil(0.3*max(H(:))));
x = T(P(:,2)); y = R(P(:,1));
plot(x,y,'s','color','white');
% Find lines and plot them
lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',4);
figure, imshow(im_right), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
end
