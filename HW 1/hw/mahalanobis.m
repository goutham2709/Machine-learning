
function [ mahal_output ] = mahalanobis(A,B,M)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
P = inv(M);
for i= 1:150;
   mahal_output(i)= (A(i,:)-B)*P*(transpose(A(i,:)-B));
 
 
end
