function [Lpred, w] = FishersLDA(Dtrain, Ltrain, Dtest, lambda)
% Binary Classification using Fisher's linear discriminant
% 1. Find the optimal data projection direction w using the training dataset 
% 2. And use Fisher's linear discriminant to make classification for testing dataset
% Note: the class labels use 1 and -1 to represent the two classes to be classified. 
 
% Inputs: 
%   Dtrain  - training dataset
%   Ltrain  - class labels of training patterns
%   Dtest   - testing patterns  
%   lambda  - penalty matrix  
%   option  - classification options

% Outputs
%   Lpred         - predicted class labels of the testing patterns
%   w             - optimal weight vector for data projection
 

% If no input for lambda, assign a defualt matrix to lambda 
if (nargin<3 || isempty(lambda))
   lambda = [0 1; 1 0];
end

%--------Fisher Linear Discriminant---------%
idx1 = find(Ltrain==1);  % the index for class 1 
idx2 = find(Ltrain==-1); % the index for class -1  
Dtrain_c1 = Dtrain(idx1, :);  % the training samples of class 1
Dtrain_c2 = Dtrain(idx2, :);  % the training samples of class -1

N_c1 = length(idx1);  % the number of samples in class 1   
N_c2 = length(idx2);  % the number of samples in class -1 

sigma1 = cov(Dtrain_c1);
mu1 = mean(Dtrain_c1);

sigma2 = cov(Dtrain_c2);
mu2 = mean(Dtrain_c2);
Sw = sigma1 + sigma2;  

%% The optimal direction w for sample projection:   
w  = inv(Sw)*(mu1-mu2)';  

%------The Projected Data-------%
Dtrain_new = Dtrain*w; % Projected training data
Ltrain_new = Ltrain;   % Training data label
Dtest_new = Dtest*w;   % Projected testing data

Dtrain_new_c1 = Dtrain_new(idx1, :);  % projected training samples of class 1
Dtrain_new_c2 = Dtrain_new(idx2, :);  % projected training samples of class -1
 

%-------------------------------------------------------------------------%
% Complete Classification on the Projected Data on One-Dimensional Space 
% Using Derived Bayesian Decision Boundary   
% if Ratio of likelihood > [(lambda12-lambda22)/(lambda21-lambda11)]*ratio of prior
% The Optimal Decision Rule, check Slides of Lecture 4 Bayesian Thoery, Page 27

Lpred = []; 
 


