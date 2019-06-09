% T - statistic % 
function k =t_dist(X,Y)
Ex =mean(X);
Ey =mean(Y);
s =Ex-Ey;
u =X-Y;
k =(abs(s))/(std(u));
end