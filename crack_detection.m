clc,clear,close all;

% Upload image from folder named "cracks"
[filename,pathname] = uigetfile('*','Choose the input image from the "cracks" folder');
im1 = imread([pathname,filename]);
scale = 600/(max(size(im1(:,:,1))));        
im1 = imresize(im1,scale*size(im1(:,:,1)));

% % Image resize
[m,n,~] = size(im1);

      Red = im1(:,:,1);
      Green = im1(:,:,2);
      Blue = im1(:,:,3);
      
      %Get histValues for each channel
      [yRed, x] = imhist(Red);
      [yGreen, x] = imhist(Green);
      [yBlue, x] = imhist(Blue);
      %Plot them together in one plot
      figure(3)
      plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');
      title('Histogram of Cover image ');  

%PLEASE INSERT A VALUE!
% USE 0 IN CASE YOU DON'T WANT THE BRIGHTNESS TO CHANGE
br = inputdlg('Enter the increased/decreased amount of brightness');
Br = str2double(br)
im = im1 + Br;
imtool(im);

%% Image processing
% Convert image from RGB to gray scale
I = rgb2gray(im);
      Red = I(:,:,1);
      %Get histValues for each channel
      [yRed, x] = imhist(Red);
      [yGreen, x] = imhist(Green);
      [yBlue, x] = imhist(Blue);
      %Plot them together in one plot
      figure(4);
      plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');
      title('Histogram of Cover image ');  

figure(2)
subplot(1,2,1)
imhist(I)
xlim([0,250]);
title('Histogram before enhancement')
imtool(I)

% Image enhancment
% First) 9*9 low pass filter
[f1,f2]=freqspace(size(I),'meshgrid');
D=100/size(I,1);
LPF = ones(9); 
r=f1.^2+f2.^2;
for i=1:9
    for j=1:9
        t=r(i,j)/(D*D);
        LPF(i,j)=exp(-t);
    end
end


% Second) applying filter
Y=fft2(double(I)); 
Y=fftshift(Y);
Y=convn(Y,LPF); 
Y=ifftshift(Y);
I_en=ifft2(Y);

% Third) blurr image
I_en=imresize(I_en,size(I)); 
I_en=uint8(I_en);
I_en=imsubtract(I,I_en);
I_en=imadd(I_en,uint8(mean2(I)*ones(size(I))));

subplot(1,2,2)
imhist(I_en)
xlim([0,250]);
title('Histogram after enhancement')
imtool(I_en)

% Segmentation of image
level = roundn(graythresh(I_en),-2); % Calculate threshold using  Otsu's method
BW = ~im2bw(I_en,level);  % Convert image to binary image using threshold
imtool(BW)
disp("The threshold value is: ")
disp(level)

% Removing noise and conecting image
BW1 = BW;

BW2 = BW1;
%BW1 = imdilate(BW1,strel('disk',i));  % dilate image
BW1 = bwmorph(BW1,'bridge',inf);      % connecting close parts
BW1 = bwmorph(BW1,'diag',inf);
BW1 = bwmorph(BW1,'close',inf);
%imtool(BW1)
BW1 = imfill(BW1,'holes');            % filling small spaces
%BW1 = imerode(BW1,strel('disk',i-1));   % erode image
%imtool(BW)
tmp = bwareafilt(BW1,1);              % get size of biggest connected shape
tmp = fix(0.05*sum(sum(tmp)));        % size considered noise
BW1  = bwareaopen(BW1,tmp);           % remove isolated pixels
%imtool(BW)
CC = bwconncomp(BW1);
if CC.NumObjects<1;end          % break the loop at convergence

B = bwboundaries(BW1); % Cracks boundaries
imtool(BW1)

%% Claculate the length of the crack using Euclidean distance
Dist = zeros(length(B),1); % Preallocation
a = Dist; 
b = Dist; % Preallocation
for i=1:length(B)
    tmp = B{i};
    D = pdist2(tmp,tmp); % Euclidean distance between each 2 points
    % Value and position of farthest 2 points
    [D,tmp] = max(D); 
    [Dist(i),b(i)] = max(D); 
    a(i) = tmp(b(i));
end


%% Showing results
x = inputdlg('Enter the area of image in square meters:',...
             'Sample', [1 50]);
A = str2double(x{:}); 
Dist = Dist*sqrt(A/(n*m)); % convert distances into meters

figure,imshow(I_en);
hold on
for i=1:length(B)
    tmp = B{i};
    plot(tmp(:,2),tmp(:,1),'r','LineWidth',2);
    plot([tmp(a(i),2),tmp(b(i),2)],[tmp(a(i),1),...
        tmp(b(i),1)],'*-b','LineWidth',2);
    text(1+0.5*sum([tmp(a(i),2),tmp(b(i),2)]),1+0.5*sum([tmp(a(i),1),...
        tmp(b(i),1)]),num2str(Dist(i)),'Color','k','FontSize',10);
end
hold off,title("Crack's Length");
warning on