function [sensitivity, specificity] = cal_senspe(Lpred, Ltrue)
% Sensitivity and Specificity are Statistical performance measures for 
% binary classification (especially popular in medical diagnostics)

% ? True positive (TP): Sick people correctly diagnosed as sick
% ? False positive (FP): Healthy people incorrectly identified as sick
% ? True negative (TN): Healthy people correctly identified as healthy
% ? False negative (FN): Sick people incorrectly identified as healthy

% Inputs: 
%  Ltrue - true label  
%  Lpred - predicted label

% Outputs: 
% Sensitivity = TP/(TP+FN), test ability to identify positive class 1
% Specificity = TN/(TN+FP), test ability to identify negative class -1

idx1 = find(Ltrue==1); 
pred1 = Lpred(idx1); 
sensitivity = length(find(pred1==1))/length(idx1); 

idx2 = find(Ltrue==-1); 
pred2 = Lpred(idx2); 
specificity = length(find(pred2==-1))/length(idx2);