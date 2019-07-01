% IE6318 HW4 - Classification Using KNN & Decision Tree
% Attribute Information:
%  1. sepal length in cm
%  2. sepal width in cm
%  3. petal length in cm
%  4. petal width in cm
%  5. class: 
%     1 -- Iris Setosa
%     2 -- Iris Versicolour
%     3 -- Iris Virginica


clear all; clc; close all; 

%-----1.Load Raw Data----------%
% Load feat and label from iris.txt in to the workspace
data = load(['iris.txt']); 
feat = data(:,1:4); % feature matrix
label = data(:,5);  % class label vector


%-----2. Prepare N-fold dataset for classification----------%
% The function divide_nfold_data.m divide samples of each class into N fold evenly
N = 5; % N-fold cross validation 
data_nfold = divide_nfold_data(feat, label, N); 


%-----3. Perform N-fold Cross-Validation using KNN Function-----------------% 
ACC_SUM = [];
C = unique(label); %extract label information from label vector
 
for opt = [1 2] 
    acc_nfold = []; 
    
    confusion_nfold = zeros(length(C), length(C));
    
    for ifold = 1:N 
       %----prepare cross-validation training and testing dataset---% 
       idx_test = ifold; % index for testing fold
       idx_train = setdiff(1:N, ifold); % index for training folds
       Dtest = []; Ltest = []; % initialize testing data and label
       Dtrain = []; Ltrain = []; % initialize testing data and label

       %---construct the training and testing dataset for the ith fold cross validatoin
       for iC = 1:length(C) 
           cl = C(iC);   
           dtest = eval(['data_nfold.class',num2str(cl), '.fold', num2str(ifold)]);
           Dtest = [Dtest; dtest]; 
           Ltest = [Ltest; cl*ones(size(dtest,1), 1)]; 

           for itr = 1:length(idx_train)
               idx = idx_train(itr); 
               dtrain = eval(['data_nfold.class',num2str(cl), '.fold', num2str(idx)]);
               Dtrain = [Dtrain; dtrain];
               Ltrain = [Ltrain; cl*ones(size(dtrain,1), 1)]; 
           end  
       end
       %---------------------------------------------------------%

       %--------------Bayesian Classification-------------------------%  
       % Bayesian classification using 3 options
         Lpred = myBayesPredict(Dtrain, Ltrain, Dtest, opt); 

       %---------------------------------------------------------%

       %---Calculate Classification Accuracy-----%
       acc = sum(Lpred==Ltest)/length(Ltest);  
       acc_nfold(ifold, 1) = acc; 

       %---Obtain Confusion Matrix based on Lpred and Ltest-----%
       confusion_i = confusionmat(Ltest, Lpred); 
       eval(['confusionmat_fold', num2str(ifold), '=confusion_i;']); 
       confusion_nfold = confusion_nfold + confusion_i; 
    end

    acc_ave = mean(acc_nfold); % average accuracy of N folds of cross validations
    ACC_SUM = [ACC_SUM; opt, acc_ave];
end




