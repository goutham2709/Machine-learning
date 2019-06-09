% 3.1 - Minkowski's Distance %
function [Mink_output] = minkowski(A,B,r)

for i=1:150
  
Mink_output(i)=(sum((abs(A(i,:)-B)).^r).^(1/r));
 
end

