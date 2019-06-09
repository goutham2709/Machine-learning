% IE6318 HW4 - Classification Using LDA Classifier 

% Breast Cancer Coimbra Data Set
% https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra  

%% Data Set Information:
% There are 10 predictors, all quantitative, and a binary dependent variable, indicating the presence or absence of breast cancer. 
% The predictors are anthropometric data and parameters which can be gathered in routine blood analysis. 
% Prediction models based on these predictors, if accurate, can potentially be used as a biomarker of breast cancer.

% Attribute Information:
%  
% Age (years) 
% BMI (kg/m2) 
% Glucose (mg/dL) 
% Insulin (µU/mL) 
% HOMA 
% Leptin (ng/mL) 
% Adiponectin (µg/mL) 
% Resistin (ng/mL) 
% MCP-1(pg/dL) 

% Labels: 
% 1=Healthy controls 
% 2=Patients
 

%% HW4 main program
clear all; clc; close all; 

%-----1.Load Raw Data----------%
% Load feat and label from Breast_Cancer_dataset.xlsx in to the workspace
data = importdata('Breast_Cancer_dataset.xlsx');
data = data.data; 
feat = data(:,1:9); % feature matrix
label = data(:,10);  % class label vector

%---------Use IRIS Dataset---------------%
% data = load(['iris.txt']); 
% feat = data(1:100,1:4); % feature matrix
% label = data(1:100,5);  % class label vector
%----------------------------------------%

% For Binary Classification, use labels of 1 & -1
idx1 = find(label==1); 
label(idx1) = -1; % Use -1 to label Healthy controls
idx2 = find(label==2); 
label(idx2) = 1; % Use 1 to label Patients


%-----2. Prepare N-fold dataset for classification----------%
N = 5; % N-fold cross validation 
data_nfold = divide_nfold_data(feat, label, N); 

%-----3. Perform N-fold Cross-Validation using KNN Function-----------------% 
C = unique(label); %extract label information from label vector
ACC_SUM = [];
 
acc_nfold = []; 
senspe_nfold =[];  
auc_nfold = []; 

for ifold = 1:N 
   %----prepare cross-validation training and testing dataset---% 
   idx_test = ifold; % index for testing fold
   idx_train = setdiff(1:N, ifold); % index for training folds
   Dtest = []; Ltest = []; % initialize testing data and label
   Dtrain = []; Ltrain = []; % initialize testing data and label

   %---construct the training and testing dataset for the ith fold cross validatoin
   for iC = 1:length(C) 
       cl = C(iC);   
       dtest = eval(['data_nfold.class',num2str(iC), '.fold', num2str(ifold)]);
       Dtest = [Dtest; dtest]; 
       Ltest = [Ltest; cl*ones(size(dtest,1), 1)]; 

       for itr = 1:length(idx_train)
           idx = idx_train(itr); 
           dtrain = eval(['data_nfold.class',num2str(iC), '.fold', num2str(idx)]);
           Dtrain = [Dtrain; dtrain];
           Ltrain = [Ltrain; cl*ones(size(dtrain,1), 1)]; 
       end  
   end
   %---------------------------------------------------------%

   %--------------LDA Classification-------------------------%  
   % Classification using the function Fisher's Linear Discrimiant Analysis (LDA) 
   lambda = [0 1; 1 0];  % lambda = [0 1; 1 0];   
   option = 2; 
   %threshold_list = -75:1:75; 
   
   % Lpred = FishersLDA(Dtrain, Ltrain, Dtest, lambda, option);
   [Lpred, w, AUC, ROC, senspe] =  FishersLDA_v2(Dtrain, Ltrain, Dtest, Ltest, lambda, option);
 
   %---------------------------------------------------------%

   %---Calculate Classification Accuracy-----%
   acc = sum(Lpred==Ltest)/length(Ltest);  
   
   %---Calculate Sensitivity & Specificity based on Lpred and Ltest-----%
   idx1 = find(Ltest==1); pred1 = Lpred(idx1); 
   sen = length(find(pred1==1))/length(idx1); 
   idx2 = find(Ltest==-1); pred2 = Lpred(idx2); 
   spe = length(find(pred2==-1))/length(idx2);

   %---Record the results----%  
   acc_nfold(ifold, 1) = acc; 
   senspe_nfold = [senspe_nfold; sen, spe];
   auc_nfold = [auc_nfold; AUC]; 
   AUC
end

acc_ave = mean(acc_nfold); % average of N folds of cross validations
senspe_ave = mean(senspe_nfold); 
auc_ave = mean(auc_nfold);

ACC_SUM = [ACC_SUM; acc_ave, senspe_ave];





