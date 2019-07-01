from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
# % Attribute Information:
# %  1. sepal length in cm
# %  2. sepal width in cm
# %  3. petal length in cm
# %  4. petal width in cm
# %  5. class:
# %     1 -- Iris Setosa
# %     2 -- Iris Versicolour
# %     3 -- Iris Virginica

# %% The function to find points to seperate dataset to N-fold
from itertools import combinations


def nfold_set(feat, N):
    # % Determin the size of each subset
    L = len(feat)  # % number of samples
    n = int(np.floor(L / N))  # % basic subset size
    rem = L % N
    a = n * np.ones(N, dtype=np.int)
    if rem != 0:
        comb = list(combinations(list(range(N)), 2))
        rs = comb[int(np.random.rand() * N)]
        for i in rs:
            a[i] += 1
    feat_nfold = {}
    start_ind = 0
    for i in range(N):
        feat_nfold["fold{}".format(i + 1)] = feat[start_ind:start_ind + a[i]]
        start_ind += a[i]
    return feat_nfold


# % This is to split a dataset into N-fold for cross-valiation purpose
# % feature: the data matrix, each row is a smaple, each column is an attribute
# % label: class label of the samples
# % N: divide dataset into N parts with equal size
def divide_nfold_data(feature, label, N):
    C = np.unique(label)
    data_nfold = {}
    for cl in C:
        cl = int(cl)
        idx = np.argwhere(label == cl)
        data = feature[idx[:, 0]]
        L = len(idx)
        feat_nfold = nfold_set(data, N)
        data_nfold["class{}".format(cl)] = feat_nfold
    return data_nfold

#% Binary Classification using Fisher's linear discriminant
# % 1. Find the optimal data projection direction w using the training dataset
# % 2. And use Fisher's linear discriminant to make classification for testing dataset
# % Note: the class labels use 1 and -1 to represent the two classes to be classified.
#
# % Inputs:
# %   Dtrain  - training dataset
# %   Ltrain  - class labels of training patterns
# %   Dtest   - testing patterns
# %   lambda  - penalty matrix
# %   option  - classification options
#
# % Outputs
# %   Lpred         - predicted class labels of the testing patterns
# %   w             - optimal weight vector for data projection
def FishersLDA(Dtrain, Ltrain, Dtest, lambda_m=None):
    Lpred=np.zeros(len(Dtest),dtype=np.int)
    #If no input for lambda, assign a defualt matrix to lambda
    if lambda_m==None:
        lambda_m=np.array([[0,1],[1,0]],dtype=np.float)

    #%--------Fisher Linear Discriminant---------%
    idx1 =np.argwhere(Ltrain == 1) #the index for class 1
    idx2 =np.argwhere(Ltrain == -1)#the index for class -1
    # idx2 = np.argwhere(Ltrain == 2)  # the index for class -1
    Dtrain_c1 =Dtrain[idx1[:, 0]] #% the training samples of class 1
    Dtrain_c2 =Dtrain[idx2[:, 0]]  #% the training samples of class -1

    N_c1 = len(idx1);  #% the number of samples in class 1
    N_c2 = len(idx2);  #% the number of samples in class -1

    sigma1 = np.cov(Dtrain_c1, rowvar=False)
    mu1 = np.mean(Dtrain_c1, axis=0)

    sigma2 = np.cov(Dtrain_c2, rowvar=False)
    mu2 = np.mean(Dtrain_c2, axis=0)
    Sw = sigma1 + sigma2

    #%% The optimal direction w for sample projection:
    w=np.dot(np.linalg.inv(Sw),np.transpose(mu1-mu2))

    #%------The Projected Data-------%
    Dtrain_new = np.dot(Dtrain,np.transpose(w)) #% Projected training data
    Ltrain_new = Ltrain   #% Training data label
    Dtest_new = np.dot(Dtest,np.transpose(w))   #% Projected testing data

    Dtrain_new_c1 = Dtrain_new[idx1[:, 0]] #% projected training samples of class 1
    Dtrain_new_c2 = Dtrain_new[idx2[:, 0]]  #% projected training samples of class -1

    #%-------------------------------------------------------------------------%
    # % Complete Classification on the Projected Data on One-Dimensional Space
    # % Using Derived Bayesian Decision Boundary
    # % if Ratio of likelihood > [(lambda12-lambda22)/(lambda21-lambda11)]*ratio of prior
    # % The Optimal Decision Rule, check Slides of Lecture 4 Bayesian Thoery, Page 27
    return Lpred



# % This is for multi-class classification using Bayesian Decision Theory
# % Function Input:
# % Dtrain: training dataset, each row is a feature vector of a training sample
# % Ltrain: class labels of training samples
# % Dtest: testing dataset
# % opt: classification options
# %      if opt==1, use Naïve Bayes
# %      if opt==2, use posterior probability as discriminant function
# %      if opt==3, use the derived formula based on multivariate normal
# %      distribution
# %
# % Function Output:
# % Lpred: predicted class labels for the testing samples in Dtest
def myBayesPredict(Dtrain, Ltrain, Dtest, opt):
    # %% 1. Use Naive Bayes Function to Make classification
    # % Assume the     features    are    independent, then    we    can    use    Naive    Bayes    for prediction
    if opt == 1:
        NB = GaussianNB()  # % construct a Naive Bayes model NB
        Lpred = NB.fit(Dtrain, Ltrain).predict(
            Dtest)  # apply the trained model NB to predict class of test samples in Dtest
        return Lpred

    # 2.    Use    the    discriminant    function    G(x) = likelihood * prior    for classification
    # In a general case with correlated features, we can assume the features
    # follows    multivariate    normal    distribution, then    we    can    use    function    "mvnpdf"
    # to     calculate    the    likelihood    P(X | Wj)    directly
    # Decision     Rule: select    the    class that maximizes P(X | Wj)P(Wj) - likelihood * prior
    if opt == 2:
        C = np.unique(Ltrain)
        G = np.zeros(shape=(len(C), len(Dtest)), dtype=np.float)
        for iC in range(len(C)):
            cl = C[iC]
            idx = np.argwhere(Ltrain == cl)
            data = Dtrain[idx[:, 0]]
            mu = np.mean(data, axis=0)  # feature mean vector
            sigma = np.cov(data, rowvar=False)  # feature covariance matrix
            P = len(idx) / len(Ltrain)
            # For each testing sample, calculate P(X|Wj)P(Wj) = likelihood of class i * prior of class i
            mvn = multivariate_normal(mu, sigma)
            for j in range(len(Dtest)):
                x = Dtest[j]
                likelihood = mvn.pdf(x)  # likelihood of the current class i
                prior = P  # prior of the current class i
                # % Record values of the discriminat function G(X)
                # % In the following matrix G, each row represent a class, and
                # % each column represent a testing sample
                G[iC, j] = likelihood * prior  # P(X|Wj)P(Wj)

        # % For each testing sample, find the index of the class that have maximum
        # % value of likelihood*prior
        pred=np.argmax(G, axis=0)
        Lpred = C[pred]
        return Lpred

    # 3. Use the derived discriminant function G(x) for classification
    # based on the the assumption of Multivariate Normal Distribution for features
    if opt == 3:
        Lpred = []
        print("Complete the model option 3.")


# from sklearn import datasets
# iris = datasets.load_iris()
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

# %-----1.Load Raw Data----------%
# % Load feat and label in to the workspace
import numpy as np

# load data from a txt file
data = np.loadtxt("processed_iris_data.txt")
feat = data[:, :4]
label = np.array(data[:, 4], dtype=np.int)

# %-----2. Prepare N-fold dataset for classification----------%
# % The function divide_nfold_data.m divide samples of each class into N fold evenly
N = 5  # % N-fold cross validation
data_nfold = divide_nfold_data(feat, label, N)

# %-----3. Perform N-fold Cross-Validation
C = np.unique(label)  # %extract label information from label vector

ACC_SUM=np.empty(shape=[0, 2], dtype=np.float)
for opt in range(1, 3):
    acc_nfold = np.zeros(N, dtype=np.float)
    confusion_nfold = np.zeros((len(C), len(C)), dtype=np.int)

    for ifold in range(N):
        # %----prepare cross-validation training and testing dataset---%
        idx_test = ifold  # % index for testing fold
        idx_train = np.setdiff1d(np.arange(N), np.array([ifold], dtype=np.int))  # % index for training folds
        Dtest = np.empty(shape=[0, 4], dtype=np.float)
        Ltest = np.empty(shape=[0, 1], dtype=np.int)  # % initialize testing data and label

        Dtrain = np.empty(shape=[0, 4], dtype=np.float)
        Ltrain = np.empty(shape=[0, 1], dtype=np.int)  # % initialize testing data and label

        # % ---construct the training and testing dataset for the ith fold cross validatoin
        for iC in C:
            Dtest = np.append(Dtest, data_nfold["class{}".format(iC)]["fold{}".format(idx_test + 1)], axis=0)
            Ltest=np.append(Ltest, iC * np.ones(
                    shape=(data_nfold["class{}".format(iC)]["fold{}".format(idx_test + 1)].shape[0], 1)))
            for idx in idx_train:
                Dtrain = np.append(Dtrain, data_nfold["class{}".format(iC)]["fold{}".format(idx + 1)], axis=0)
                Ltrain = np.append(Ltrain, iC * np.ones(
                    shape=(data_nfold["class{}".format(iC)]["fold{}".format(idx + 1)].shape[0], 1)))

        # % --------------Bayesian Classification - ------------------------ %
        # % Bayesian classification using 3 options
        Lpred = myBayesPredict(Dtrain, Ltrain, Dtest, opt)

        #%---Calculate Classification Accuracy-----%
        acc = sum(Lpred == Ltest) / len(Ltest)
        acc_nfold[ifold] = acc
        #%---Obtain Confusion Matrix based on Lpred and Ltest-----%
        confusion_i = confusion_matrix(Ltest, Lpred)
        confusion_nfold+=confusion_i

    acc_ave = np.mean(acc_nfold) #average accuracy of N folds of cross validations
    ACC_SUM= np.append(ACC_SUM, np.array([[opt,acc_ave]],dtype=np.float), axis=0)


