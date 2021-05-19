%To test each function we can use these testing template code outside of
%this class and acces each function with dot operation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
%     
% clc;    
% clear;
% close all;
%  %Load the massspec data
%  massspec_data=load('massspec_data.mat');
%  X=massspec_data.X; 
%  
%  %call pca_dimred function
% [EigenVector_of_masspec,eigen_value_massspec]= txy822_assignment2_part2.pca_dimred(X);
% 
% 
% %load the data set
% %Assuming the data is available in the same folder with this file
% face_data=load('faces.mat'); 
% 
% %input images 
% input_images= face_data.raw_images;
% 
% %call Eigen_Faces function
% [eigenfaces]= txy822_assignment2_part2.Eigen_Faces(input_images);
% 
% %Three test images
% input_test=input_images{29};
% test_image1= rgb2gray(imread ('Sunglasses.png'));
% test_image2=rgb2gray(imread('Celeb.png'));
% 
% txy822_assignment2_part2.Facial_recognition(resized_test_image2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    



classdef txy822_assignment2_part2
    
   
    
methods(Static)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    
% q part2_1   
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%  function to pca of massspec data   
function [W,D]=pca_dimred(X)
    
%to use this function alone uncomment this
%Load the massspec data
%massspec_data=load('massspec_data.mat');
%X=massspec_data.X; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculate min of data
mean_x=mean(X);

%zero mean matrix
shift_x=X-mean_x;
%normalization
Xnorm = (shift_x) ./ std(X,1);

%
Xstd= std(Xnorm,1);

%covariance matrix
covariance_matrix=cov(Xnorm);

% V: eigenvector matrix  U : eigenvalue matrix
[V,U]= eig(covariance_matrix);

% Format the eigen value and eigen vector matrix
% with diagonal matrix D to 1XN  matrix
W=[];
D=[];
for i=1:size(V,2)
        W=[W V(:,i)];
        D=[D U(i,i)];

 end
 
 %sort the eigen value matrix 
 [B index]=sort(D);
 ind=zeros(size(index));
 dtemp=zeros(size(index));
 wtemp=zeros(size(W));
 len=length(index);
 for i=1:len
    dtemp(i)=B(len+1-i);
    ind(i)=len+1-index(i);
    wtemp(:,ind(i))=W(:,i);
 end
 D=dtemp;
 W=wtemp;
 
%Top 10 eigen values in ascending order
 top_10_eigen_values=maxk(D,10);
  top_10_eigen_values;
 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%q part2_2 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [eigenfaces]= Eigen_Faces(input_images)
    
%we have M images
[~,M]=size(input_images);
j=10;%number of eigen faces

[irow,icol]=  size(input_images{1});

%matrix to hold all images colomnwize of  each image by concatinating 
image_matrix = [];
for i = 1 : M
    img=input_images{i};
    [r c] = size(img);            %considering all images has the same size
    temp = reshape(img',r*c,1);
    image_matrix = [image_matrix temp];
end

%mean of the matrix 
 m = mean(image_matrix,2); 
 
%normalize the image matrix
 mean_x=mean(image_matrix);
p=double(image_matrix);
%zero mean matrix
shift_x=double(image_matrix)-mean_x;
%normalization
Xnorm = (shift_x) ./ std(p,1);


%using the biult in pca function we can get V eigen vector, D eigen values
[V,score,D,~,explained,~]=pca(Xnorm);

%the total variance can be calculated from the sum of latent or eigen
%values
total_variance = sum(D);

%Once we get the explaned we can  plot the graph of cumulative variance vs
%number of principal
%components for n=5,10,20,50 and 86 as follows

N = [5 10 20 50 86];

for i=1:size(N,2)
figure;
explained_n=maxk(explained,N(i));
plot(1:N(i),cumsum(explained_n));
xlabel('Principal Component');
ylabel('Cumulative Variance (%)');
title(['Cumulative Variance for first ' num2str(N(i)) ' PCs']);  
if(N(i)==86)
 title('Cumulative Variance for first ALL PCs'); 
else
    title(['Cumulative Variance for first ' num2str(N(i)) ' PCs']);  
    
end


    
end

% to find eigen faces  and display top 10 based on eigen value
eigenfaces=Xnorm*V;
figure(j);
for i=1:j      %j=number of eigen faces 
    img=reshape(eigenfaces(:,i),icol,irow);
    img=img';
    img=histeq(img,255);
    subplot(ceil(sqrt(j)),ceil(sqrt(j)),i)
    imshow(img)
    title(['Eigen Face = ' num2str(i) ' '])
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%q part2_3 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%the test image for this function is gray or NXM or other but not RGB
function  Facial_recognition(test_image)
 %load data
 face_data=load('faces.mat'); 
%input images 
input_images= face_data.raw_images;
[~,M]=size(input_images);

[irow,icol]=  size(input_images{1}); %assume all images has equal size 
% test_image=rgb2gray(test_image);                     %with image 1
% resized_test_image=imresize(test_image,[irow,icol]);%adjust the shape of
%                                                     %test image


%threshold to block unseen images. Take  minimum euclidian distance
threshold=3.89e+13;


%matrix to hold all images colomnwize of  each image by concatinating 
image_matrix = [];
for i = 1 : M
    img=input_images{i};
    [r c] = size(img);            %considering all images has the same size
    temp = reshape(img',r*c,1);
    image_matrix = [image_matrix temp];
end

%mean of the matrix 
 m = mean(image_matrix,2); 
 
%normalize the image matrix
normalized_matrix = [];
for i=1 : M
    im1=double(image_matrix(:,i));
    temp = im1 - m;
    normalized_matrix = [normalized_matrix temp];
end

%using the biult in pca function we can get V eigen vector, D eigen values
[V,score,D,~,explained,~]=pca(normalized_matrix);

eigenfaces=normalized_matrix*V;
projectimg = [ ];  % projected image vector matrix
for i = 1 : size(eigenfaces,2)
    temp = eigenfaces' * normalized_matrix(:,i);% for each column
    projectimg = [projectimg temp];
end
% extractiing PCA features of the test image %%%%%
test_image = test_image(:,:,1);% read each value row by col or convert to number value
[r c] = size(test_image);
temp = reshape(test_image',r*c,1); % creating (MxN)x1 image vector from the 2D image
size(temp)
size(m)
temp = double(temp)-m; % mean subtracted vector

projtestimg = eigenfaces'*temp; % projection of test image onto the facespace

% calculating & comparing the euclidian distance of all projected 
%trained images from the projected test image %%%%%
 euclide_dist = [ ];
 for i=1 : size(eigenfaces,2)
    temp = (norm(projtestimg-projectimg(:,i)))^2;
    euclide_dist = [euclide_dist temp];
 end
[euclide_dist_min recognized_index_min] = min(euclide_dist);
[euclide_dist_max recognized_index_max] = max(euclide_dist);

%placing threshold to filter out uknown or un seen images 
if(euclide_dist_min < threshold)

    img=test_image;
    subplot(1,2,1);
    imshow(img)
    title('Input Face');
    img=input_images{recognized_index_min};
    subplot(1,2,2);
    imshow(img)
    title('Recognized Face :ACCESS GRANTED');
   
else
    img=test_image;
    subplot(1,2,1);
    imshow(img)
    title('Input Face');
    img=zeros([60,50]);
    % im=imread('access_denied.png')
    subplot(1,2,2);
    imshow(img)
    title('ACCESS DENIED');
    
    
end

end

   end

end