% IE6318 HW1 - IRIS Dataset Study  
% Attribute Information:
%  1. sepal length in cm
%  2. sepal width in cm
%  3. petal length in cm
%  4. petal width in cm
%  5. class: 
%     1 -- Iris Setosa
%     2 -- Iris Versicolour
%     3 -- Iris Virginica
      
clear all; % before run the main program, clear all the variables in the workspace 
clc; % clears all the text from the Command Window, resulting in a clear screen.
close all; % close all the previously generated figures before starting a new run


%-----Load Raw Data----------%
%Load feat and label from iris.txt in to the workspace
data = load('iris.txt') 
feat = data(:,1:4); % feature matrix
label = data(:,5);  % class label vector

 
%-----Explore Iris Dataset----------%
%2.1 2D scatter plot of four attributes
gplotmatrix(feat, feat, label); 
title('2D pairwise scatter plot of the attributes', 'fontsize', 16, 'fontweight', 'bold'); 

%2.2 3D scatter plot of three attributes 
figure;
idx1 = find(label==1); 
X1 = feat(idx1,1); 
Y1 = feat(idx1,2); 
Z1 = feat(idx1,3); 
scatter3(X1(:),Y1(:),Z1(:)); %view(-60,60), view(40,35) 
hold on; 

idx2 = find(label==2);  
X2 = feat(idx2,1); 
Y2 = feat(idx2,2); 
Z2 = feat(idx2,3); 
scatter3(X2(:),Y2(:),Z2(:)); %view(-60,60), view(40,35)
hold on; 

idx3 = find(label==3); 
X3 = feat(idx3,1); 
Y3 = feat(idx3,2); 
Z3 = feat(idx3,3); 
scatter3(X3(:),Y3(:),Z3(:), 'filled'); %view(-60,60), view(40,35)

title('3D Scatter Plots of Three Features');
xlabel('sepal length');  
ylabel('sepal width');  
zlabel('petal length');

%2.3 3D scatter plot of three attributes
figure; 
imagesc(feat); colorbar; 

%2.4 Histogram of the attributes values
figure; 
histogram(X1,'FaceColor','g'); 
hold on;  
histogram(X2,'FaceColor','r'); 
ylabel('counts')
xlabel('data range');

%2.5 Correlation Matrix and visualize 
figure; 
C = corr(feat); 
imagesc(C); colorbar; 

%2.6 Parallel coordinates plot
figure;  
parallelcoords(feat, 'group', label); 

%2.7 Boxplot
figure;
boxplot(feat(:,1),label);
boxplot(feat(:,2),label);
boxplot(feat(:,3),label);
boxplot(feat(:,4),label);


%3: Make functions for Minkowski Distance,  T-statistics Distance, Mahalanobis Distance


%4.1 Calculate Minkowski distances
figure;
A = feat(:,:);
B = [5.0000,3.5000,1.4600,0.2540];
r = 1;
Mink_output= minkowski(A,B,r);
plot(Mink_output);


figure;
A = feat(:,:)
B = [5.0000,3.5000,1.4600,0.2540];
r = 2;
Mink_output= minkowski(A,B,r);
plot(Mink_output);

figure;
A = feat(:,:);
B = [5.0000,3.5000,1.4600,0.2540];
r = 100;
Mink_output= minkowski(A,B,r);
plot(Mink_output);


%4.2 Calculate Mahalanobis distances 
figure;
A = feat(:,:);
B = [5.0000,3.5000,1.4600,0.2540];
M = cov(A);
mahal_output = mahalanobis(A,B,M);
plot(mahal_output);


%5.1 import Timeseries in workspace
Timeseries = load(['time_series.txt']); 

figure;
T1 = Timeseries(:,1);
T2 = Timeseries(:,2);
plot(T1);
hold on;
plot(T2);

%5.2 T-statistics for the two time series 
Tdis = tstatistics(T1,T2)

%5.3 Correlation of the two time series data
CR = corr(T1,T2);

%5.4: feature matrix normalization
% z-scores = (X-MEAN(X)) ./ STD(X)
Z = zscore(feat);




