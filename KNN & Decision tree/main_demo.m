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

%-----1.Load Raw Data----------%
% Load feat and label from iris.txt in to the workspace
data = heartdiseasedata; 
feat = data(:,1:13); % feature matrix
label = data(:,14);  % class label vector
C = unique(label); %extract label information from label vector

Dtrain = feat([1:100,175:275], :);
Ltrain = label([1:100,175:275], :);
Dtest = feat([101:174,276:303], :);
 
opt = 2; 
Lpred = myBayesPredict(Dtrain, Ltrain, Dtest, opt) 
 


