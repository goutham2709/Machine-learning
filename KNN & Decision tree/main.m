Classification Using KNN & Decision Tree
% Attribute Information:
%  1. sepal length in cm
%  2. sepal width in cm
%  3. petal length in cm
%  4. petal width in cm
%  5. class: 
%     1 -- Iris Setosa
%     2 -- Iris Versicolour
%     3 -- Iris Virginica



%-----Load Raw Data----------%
%Load feat and label from iris.txt in to the workspace
data = heartdiseasedata; 
feat = data(:,1:13); % feature matrix
label = data(:,14);  % class label vector

%1. KNN Classification  
N = 5; % N-fold cross validation 
C = unique(label); %extract label information from label vector

% Prepare Training Dataset into N-fold 
data_nfold = divide_nfold_data(feat, label, N); 

%------Classification-----------------% 
%1. KNN Classification 
for K = [3 5 7] 
for Dorder = [1 2 5] 
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

       %--------------KNN classification-------------------------%  
       % Make a KNN function to Perform KNN classification 
       % Lpred = myknn(Dtrain, Ltrain, Dtest, K, Dorder); 
       
       % Assume Lpred = Ltest to make the demo code to run
         Lpred = Ltest;  % predicted label for testing dataset
       %---------------------------------------------------------%

       %---Calculate Classification Accuracy-----%
       acc = sum(Lpred==Ltest)/length(Ltest);  
    end  
end
end

