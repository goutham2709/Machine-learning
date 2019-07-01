function [Lpred, w, AUC, ROC, senspe] = FishersLDA_v2(Dtrain, Ltrain, Dtest, Ltest, lambda, option)
% Binary Classification using Fisher's linear discriminant
% 1. Find the optimal data projection direction w using the training dataset 
% 2. And use Fisher's linear discriminant to make classification for testing dataset
% Note: the class labels use 1 and -1 to represent the two classes to be classified. 
 
% Inputs: 
%   Dtrain  - training dataset
%   Ltrain  - class labels of training patterns
%   Dtest   - testing data  
%   Ltest   - testing data label 
%   lambda  - penalty matrix  
%   option  - classification options

% Outputs
%   Lpred         - predicted class labels of the testing patterns
%   w             - optimal weight vector for data projection
 

%% Setup the Inputs Parameters
% If no input for lambda, assign a defualt matrix to lambda 
if (nargin<5 || isempty(lambda))
   lambda = [0 1; 1 0];
   option = 3; 
end

if (nargin<6)
   option = 3;
end

%--------------------------------------%

 

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
 

mu1_new = mean(Dtrain_new_c1);  % mean of projected samples of class 1
mu2_new = mean(Dtrain_new_c2);  % mean of projected samples of class -1
% sigma1_new = cov(Dtrain_new_c1); 
% sigma2_new = cov(Dtrain_new_c2);
sigma1_new = std(Dtrain_new_c1);  % standard deviation of projected samples of class 1
sigma2_new = std(Dtrain_new_c2);  % standard deviation of projected samples of class -1
 

%%--------------------------------------------------------------%
% Classification on the Projected Data on One-Dimensional Space
Ntest = size(Dtest, 1);

if option==1
    %% Option 1: Using Derived Bayesian Decision Rule
    % if Ratio of likelihood > [(lambda12-lambda22)/(lambda21-lambda11)]*ratio of prior
    % The Optimal Decision Rule, check Slides of Lecture 4 Bayesian Thoery, Page 27
    threshold = 0; 
    Lpred = [];
    Vlist = []; 
    for i = 1:Ntest
        feat = Dtest_new(i); 

        prior1 = length(idx1)/length(Ltrain);
        likelihood1 = normpdf(feat, mu1_new,sigma1_new); % likelihood of the current class 1
        prior2 = length(idx2)/length(Ltrain);
        likelihood2 = normpdf(feat, mu2_new,sigma2_new); % likelihood of the current class -1

        Vcheck = (likelihood1/likelihood2) - (lambda(1,2)-lambda(2,2))/(lambda(2,1)-lambda(1,1))*(prior2/prior1); 
        if Vcheck > threshold
           pred = 1;
        else
           pred = -1;
        end

        Lpred(i,1) = pred; 
        Vlist(i,1) = Vcheck; 
    end
end


if option==2
    %% Option 2: Using the mid-line of projected means
    %threshold_star = (mu1_new + mu2_new)/2;
    threshold = 0;
    Lpred = [];
    Vlist = []; 
 
    for i = 1:Ntest
        feat = Dtest_new(i);       
        Vcheck = abs(feat-mu1_new) - abs(feat-mu2_new); 
        if Vcheck < threshold
           pred = 1;
        else
           pred = -1;
        end

        Lpred(i,1) = pred; 
        Vlist(i,1) = Vcheck; 
    end
end


if option==3
    %% Option 3: Using the myBayesPredict funciton made in HW3
    Lpred = myBayesPredict(Dtrain_new, Ltrain_new, Dtest_new, 3);
end



%% Perform ROC Analysis for Option 1
if option==1 
    %% Option 1: Using Derived Bayesian Decision Boundary
    % if Ratio of likelihood > [(lambda12-lambda22)/(lambda21-lambda11)]*ratio of prior
    % The Optimal Decision Rule, check Slides of Lecture 4 Bayesian Thoery, Page 27
    senspe = []; 
    bin = range(Vlist)/150; 
    Vlist_sort = sort(Vlist, 'ascend');
    threshold_list = (Vlist_sort(1)-bin):bin:(Vlist_sort(end)+bin);
    
    for ilist = 1:length(threshold_list)
        threshold = threshold_list(ilist); 
        Lpred = [];
        for i = 1:Ntest
            feat = Dtest_new(i); 

            prior1 = length(idx1)/length(Ltrain);
            likelihood1 = normpdf(feat, mu1_new,sigma1_new); % likelihood of the current class 1
            prior2 = length(idx2)/length(Ltrain);
            likelihood2 = normpdf(feat, mu2_new,sigma2_new); % likelihood of the current class -1

            Vcheck = (likelihood1/likelihood2) - (lambda(1,2)-lambda(2,2))/(lambda(2,1)-lambda(1,1))*(prior2/prior1); 
            if Vcheck > threshold
               pred = 1;
            else
               pred = -1;
            end

            Lpred(i,1) = pred; 
        end
        
        [sen, spe] = cal_senspe(Lpred, Ltest); 
        senspe = [senspe; sen spe];     
    end
    
    
    
    %% To calculate ROC 
    % ROC = [senspe(:,1), 1-senspe(:,2)];
    sen = senspe(:,1); 
    if sen(1) > sen(end)
       senspe = senspe(end:-1:1, :);
    end
    ROC = [senspe(:,1), 1-senspe(:,2)];  
    bin = ROC(:,2) - [0; ROC(1:end-1,2)]; 
    AUC = sum(ROC(:,1).*bin);  
    plot(ROC(:,2), ROC(:,1));  
    xlim([-0.2 1.2]);
    ylim([0 1.2]);
end


%% Perform ROC Analysis for Option 2
if option==2 
    %% Option 2: Using the mid-line of projected means
    senspe = [];
    bin = range(Vlist)/150; 
    Vlist_sort = sort(Vlist, 'ascend');
    threshold_list = (Vlist_sort(1)-bin):bin:(Vlist_sort(end)+bin);
    
    for ilist = 1:length(threshold_list)
        threshold = threshold_list(ilist);   
        Lpred = [];
        for i = 1:Ntest
            feat = Dtest_new(i); 
            if abs(feat-mu1_new) - abs(feat-mu2_new) < threshold
               pred = 1;
            else
               pred = -1;
            end

            Lpred(i,1) = pred; 
        end
        
        [sen, spe] = cal_senspe(Lpred, Ltest); 
        senspe = [senspe; sen spe];  
    end
    
    %% To calculate ROC 
    % ROC = [senspe(:,1), 1-senspe(:,2)];
    sen = senspe(:,1); 
    if sen(1) > sen(end)
       senspe = senspe(end:-1:1, :);
    end
    ROC = [senspe(:,1), 1-senspe(:,2)];  
    bin = ROC(:,2) - [0; ROC(1:end-1,2)]; 
    AUC = sum(ROC(:,1).*bin);  
    plot(ROC(:,2), ROC(:,1));  
    xlim([-0.2 1.2]);
    ylim([0 1.2]);
end


%% ROC Analysis for Option 3 is not available
if option==3 
   disp('This option is not available now.');
   ROC = [];
   AUC = []; 
   senspe = []; 
end



