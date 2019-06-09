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
idx2 = find(Ltrain==2); % the index for class 2  
Dtrain_c1 = Dtrain(idx1, :);  % the training samples of class 1
Dtrain_c2 = Dtrain(idx2, :);  % the training samples of class 2

N_c1 = length(idx1);  % the number of samples in class 1   
N_c2 = length(idx2);  % the number of samples in class 2 

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
Dtrain_new_c2 = Dtrain_new(idx2, :);  % projected training samples of class 2

mu1_new = mean(Dtrain_new_c1);  % mean of projected samples of class 1
mu2_new = mean(Dtrain_new_c2);

sigma1_new = std(Dtrain_new_c1);  % standard deviation of projected samples of class 1
sigma2_new = std(Dtrain_new_c2);

Ntest = size(Dtest, 1);
Lpred = [];

for i = 1:Ntest
        feat = Dtest_new(i);
        prior1 = length(idx1)/length(Ltrain);
        likelihood1 = normpdf(feat, mu1_new,sigma1_new); % likelihood of the current class 1
        prior2 = length(idx2)/length(Ltrain);
        likelihood2 = normpdf(feat, mu2_new,sigma2_new); % likelihood of the current class 2

        if (likelihood1/likelihood2) > (lambda(1,2)-lambda(2,2))/(lambda(2,1)-lambda(1,1))*(prior2/prior1); 
           pred = 1;
        else
           pred = 2;
        end

        Lpred(i,1) = pred;  
    end
end

%-------------------------------------------------------------------------%
% Complete Classification on the Projected Data on One-Dimensional Space 
% Using Derived Bayesian Decision Boundary   
% if Ratio of likelihood > [(lambda12-lambda22)/(lambda21-lambda11)]*ratio of prior
% The Optimal Decision Rule, check Slides of Lecture 4 Bayesian Thoery, Page 27
 
  
 


