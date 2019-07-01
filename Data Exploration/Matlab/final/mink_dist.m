% 3.1 - Minkowski's Distance %
function E =minko_dist(A,b,n,r)
p =size(A,1);
q =size(b,1);
for i=1:p
    for j=1:q
        for k=1:n
            F =A(i)- b(j);
            G =F^r;
            H =sum(G);
            E(i,1) =H^(1/r);
        end
    end
end
end