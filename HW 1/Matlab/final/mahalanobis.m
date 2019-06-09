% 3.3 - Mahalanobis Distance %
function d =mahanalobisdistance234(A,b)
[p,k1] =size(A);
[q,k2] =size(b);
n =p+q;
if(k1~=k2)
    disp('no of columns in A and b must be same')
else
 for i=1:p
     for j=1:q
    xDiff =A(i,1:4)-b(j,1:4);
    ca =cov(A);
    cb =cov(b);
    zc =p/n*ca+q/n*cb;
    d(i,1) =sqrt((xDiff)*inv(zc)*(xDiff)');
     end
 end
end