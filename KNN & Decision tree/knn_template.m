% This is the sample codes to find K nearest neighbor for one test sample
% from a training dataset with 150 samples and 3 classes

Data_class1 = rand(50, 4)+0.3;
Label1 = ones(50,1); 
Data_class2 = rand(50, 4);
Label2 = 2*ones(50,1);
Data_class3 = rand(50, 4)-0.3;
Label3 = 3*ones(50,1); 

% Generate Sample Training Dataset with 3 classes
Dtrain = [Data_class1; Data_class2; Data_class3];
Ltrain = [Label1; Label2; Label3];

% Feature Vector of a Testing Sample
Ftest = rand(1, 4) +0.5;

% Task: find K Nearest Neighbor in the training dataset, and do majority
% voting for classificaiton on the testing sample 
 
% K: the number of neighbors needed for classification
K = 5; 
 
% Dorder: we use Minkowski Distance   
Dorder = 2; 

%% The following program is to find the KNN for the testing sample 

% Step 1: Find class labels from label vector Ltrain
C = unique(Ltrain); % label representations, such as [-1, 1], [1, 2, 3]
r = Dorder; % order of Minkowski distance 
Lpred = []; % Initialize the vector for predicted lables for testing dataset Dtrest


% Step 2: Calculate distance between Ftest and Samples in the training dataset
Ns = size(Dtrain, 1); % # of samples in the training dataset
dmat = abs(Dtrain-repmat(Ftest, Ns, 1));   
dlist = nthroot(sum(dmat.^r, 2), r); % Calculated Minkowski Distances with order of r

% Step 3: sort the distance vector in Ascending Order
[dsort, isort] = sort(dlist, 'ascend');

% Step 4: find class labels for the K-nearest neighbors 
Lknn = Ltrain(isort(1:K)); 

% Step 5: in the K nearest neighbors, count how many neighbors from each class 
Ncl = []; % the matrix to store the count number of each class 
for iC = 1:length(C) % loop for the three classes   
    cl = C(iC); 
    ncl = length(find(Lknn==cl)); % find how many belong to class 'cl' 
    % store count number and class index into matrix Ncl   
    Ncl = [Ncl; ncl, cl]; % The 1st column records # of samples, the 2nd column records class label
end
 
% Step 6: find which class has the most samples in the K neighbors 
[vmax, imax] = max(Ncl(:,1)); % find which class has most samples in K neighbors 

% Step 7: predict the class of the test sample to the majority class in K neighbors
Cpred = Ncl(imax, 2); 

Cpred



