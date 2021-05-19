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