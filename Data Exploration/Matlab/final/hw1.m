clear
clc

% Load Data %
data = importdata('iris.txt');
features = data(:,1:4);
class = data(:,5);

% 2.1 - 2D Scatter plot matrix %
figure(1);
plotmatrix(features)

% 2.2 - 3D Scatter plot of 3 attributes %
Sepal_length = features(:,1);
Sepal_width = features(:,2);
Petal_width = features(:,4);
figure(2);
scatter3(Sepal_length, Sepal_width, Petal_width)

% 2.3 - Visualization of the feature matrix (4 columns) %
figure(3);
imagesc(features)

% 2.4 - Histogram of four attribues of 3 classes %
figure; 
histogram(Sepal_length,'FaceColor','g'); 
hold on;  
histogram(Sepal_width,'FaceColor','r'); 
ylabel('counts')
xlabel('data range');

% 2.5 - Boxplots %
figure(4)
boxplot(features(:,1),class);
figure(5)
boxplot(features(:,2),class);
figure(6)
boxplot(features(:,3),class);
figure(7)
boxplot(features(:,4),class);

% 2.6 - Correlation matrix and plot %
figure(8) 
C = corr(features)
imagesc(C)
colorbar;

% 2.7 - Parallel coordinates plot %
figure(9)
parallelcoords(features,'group', class)

% 3.1 - Minkowski's distance %
figure(10);
A = features(:,:);
B = [5.0000,3.5000,1.4600,0.2540];
r = 1;
mink_out= minkowski(A,B,r);
plot(mink_out);

figure(11);
A = features(:,:)
B = [5.0000,3.5000,1.4600,0.2540];
r = 2;
mink_out= minkowski(A,B,r);
plot(mink_out);

figure(12);
A = features(:,:);
B = [5.0000,3.5000,1.4600,0.2540];
r = 100;
mink_out= minkowski(A,B,r);
plot(mink_out);
% 3.2 -  T-statistics Distance %



% 5.1 - Import timeseries data %
Timeseries = load(['timeseries.txt']); 

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


