function H = myknn(A,B,C,r,k)
% A training data set %
%B test data set%
%C label of traning data set%
p =size(A,1);
q =size(B,1);
for z= 1:q
    D =abs(minko_dist123(A,B(z,:),r));
    [F,I]= sort(D);
    G = I(2:k+1,1);
     V = C(G,1);
    m=sum(V==1);
    s= sum(V==2);
    l= sum(V==3);
    if (m>=s) && (m>=l) 
        H(z,1) =1;
    elseif (s>=m) && (s>=l)
        H(z,1) =2;
    else(l>=m) && (l>=s);
        H(z,1) =3;
    end
end
