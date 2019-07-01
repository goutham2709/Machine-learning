function data_nfold = divide_nfold_data(feature, label, N)
% This is to split a dataset into N-fold for cross-valiation purpose
% feature: the data matrix, each row is a smaple, each column is an attribute
% label: class label of the samples
% N: divide dataset into N parts with equal size

C = unique(label); 
for iC = 1:length(C)
    cl = C(iC); 
    idx = find(label==cl); 
    data = feature(idx,:);
    L = length(idx);     
    feat_nfold = nfold_set(data, N); 
    eval(['data_nfold.class', num2str(iC), '=feat_nfold;']);     
end


%% The function to find points to seperate dataset to N-fold 
function feat_nfold = nfold_set(feat, N)
% Determin the size of each subset
L = size(feat,1); % number of samples
n = floor(L/N);   % basic subset size 
rem = mod(L, N);  % Modulus after division, there are some extra samples to assign
a = n*ones(N,1);
if rem>0
  b = nchoosek(1:N,rem);
  c = ceil(rand*size(b,1));
  idx = b(c,:); 
  a(idx)= a(idx) + 1; 
end 
nfoldpt =[0; cumsum(a)];
nint = [nfoldpt(1:end-1)+1, nfoldpt(2:end)];  

for i = 1:N
    dsub = feat(nint(i,1):nint(i,2), :); 
    eval(['feat_nfold.fold', num2str(i), '=dsub;']); 
end





