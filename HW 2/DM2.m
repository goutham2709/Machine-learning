data = importdata('iris.txt');
training = data([1:40,51:90,101:140],1:4);
trainingL = data([1:40,51:90,101:140],5);
testing = data([41:50, 91:100, 141:150], 1:4);
testingL = data([41:50, 91:100, 141:150], 5);
%size(training)
%size(testing)

Dtrain = [Data_class1; Data_class2; Data_class3];
Ltrain = [Label1; Label2; Label3];

%%KKN
for k = [3 5 7]
    for r =  [1 2 5]
        for i = 1:30
            A = training;
            B = testing(i,:);
            dist = minkowski(A, B, r);
            [sorted, index] = sort(dist);
            knnindex = index(:,1:k).';
            knnclass = trainingL(knnindex);
            pred(i, :) = mode(knnclass);
    
        end
        confu = confusionmat(testingL, pred)
        x = 0;
        for i = 1:30
            if pred(i) == testingL(i)
                x = x+1;
            end
        end
        acc = x/30*100
        x = 0;
        for i = 1:10
            if pred(i) == testingL(i)
                x = x+1;
            end
        end
        acc1 = x/10*100
        x = 0;
        for i = 11:20
            if pred(i) == testingL(i)
                x = x+1;
            end
        end
        acc2 = x/10*100
        x = 0;
        for i = 21:30
            if pred(i) == testingL(i)
                x = x+1;
            end
        end
        acc3 = x/10*100
    end
end

%Decision tree
function z = decissiontree(C,D)
p =size(C,1);
for i=1:p
    if D(i,1) >= 1.7 && C(i,1)>4.3
    z(i,1) =3;
    else
    z(i,1) =2;
    end
    end
end
%
for K = [3 5 7] 
for Dorder = [1 2 5] 
    for ifold = 1:N 
       %----prepare cross-validation training and testing dataset---% 
       idx_test = ifold; % index for testing fold
       idx_train = setdiff(1:N, ifold); % index for training folds
       Dtest = []; Ltest = []; % initialize testing data and label
       Dtrain = []; Ltrain = []; % initialize testing data and label
       
       %---construct the training and testing dataset for the ith fold cross validatoin
       for iC = 1:length(C) 
           cl = C(iC);   
           dtest = eval(['data_nfold.class',num2str(iC), '.fold', num2str(ifold)]);
           Dtest = [Dtest; dtest]; 
           Ltest = [Ltest; cl*ones(size(dtest,1), 1)]; 

           for itr = 1:length(idx_train)
               idx = idx_train(itr); 
               dtrain = eval(['data_nfold.class',num2str(iC), '.fold', num2str(idx)]);
               Dtrain = [Dtrain; dtrain];
               Ltrain = [Ltrain; cl*ones(size(dtrain,1), 1)]; 
           end  
       end
       %---------------------------------------------------------%

       %--------------KNN classification-------------------------%  
       % Make a KNN function to Perform KNN classification 
       % Lpred = myknn(Dtrain, Ltrain, Dtest, K, Dorder); 
       
       % Assume Lpred = Ltest to make the demo code to run
         Lpred = Ltest;  % predicted label for testing dataset
       %---------------------------------------------------------%

       %---Calculate Classification Accuracy-----%
       acc = sum(Lpred==Ltest)/length(Ltest);  
    end  
end
end