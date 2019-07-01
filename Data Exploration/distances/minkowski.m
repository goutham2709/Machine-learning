function [Mink_output] = minkowski(A,B,r)
%   Detailed explanation goes here
for i=1:150
    
Mink_output(i)=(sum((abs(A(i,:)-B)).^r).^(1/r));
 
end

