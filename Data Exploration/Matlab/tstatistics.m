function [tstat] = tstatistics(A,B)
%UNTITLED2 Summary of this function goes here
%  Detailed explanation goes here
for i=1:100
    
tstat = ((abs(mean(A)-mean(B)))/std(A-B));
 
end

