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