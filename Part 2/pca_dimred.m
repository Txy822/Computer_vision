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