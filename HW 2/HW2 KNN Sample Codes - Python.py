# % This is the sample codes to find K nearest neighbor for one test sample
# % from a training dataset with 150 samples and 3 classes

import numpy as np

Data_class1=np.random.uniform(0,1.0,(50,4))+0.3
Label1 = np.ones((50,1),dtype=np.float32)

Data_class2 =np.random.uniform(0,1.0,(50,4))
Label2 = np.ones((50,1),dtype=np.float32)*2

Data_class3 =np.random.uniform(0,1.0,(50,4))-0.3
Label3 = np.ones((50,1),dtype=np.float32)*3

# % Generate Sample Training Dataset with 3 classes
Dtrain = np.append(Data_class1,Data_class2,axis=0)
Dtrain=np.append(Dtrain,Data_class3,axis=0)
Ltrain = np.append(Label1,Label2,axis=0)
Ltrain=np.append(Ltrain,Label3,axis=0)

# % Feature Vector of a Testing Sample
Ftest = np.random.uniform(0,1,(1,4))+0.5

# % Task: find K Nearest Neighbor in the training dataset, and do majority
# % voting for classificaiton on the testing sample

# % K: the number of neighbors needed for classification
K=5

# % Dorder: we use Minkowski Distance
Dorder = 2




## %% The following program is to find the KNN for the testing sample

# % Step 1: Find class labels from label vector Ltrain
C = np.unique(Ltrain) #% label representations, such as [-1, 1], [1, 2, 3]
r = Dorder #% order of Minkowski distance
Lpred = [] #% Initialize the vector for predicted lables for testing dataset Dtrest

# % Step 2: Calculate distance between Ftest and Samples in the training dataset
Ns =Dtrain.shape[0]   # % of samples in the training dataset
dmat=np.absolute(np.subtract(Dtrain,Ftest))
dlist =np.power(np.sum(np.power(dmat,r),axis=1),1.0/r)# % Calculated Minkowski Distances with order of r


# % Step 3: sort the distance vector in Ascending Order
isort=dlist.argsort()
dsort=dlist[isort]

# % Step 4: find class labels for the K-nearest neighbors
Lknn = np.array(Ltrain[isort[0:K]],dtype=np.int32)

# % Step 5: in the K nearest neighbors, count how many neighbors from each class
 #% to store the count number of each class
unique, counts = np.unique(Lknn, return_counts=True)
Ncl=dict(zip(unique, counts))

# % Step 6: find which class has the most samples in the K neighbors
# max(Ncl.items(), key = lambda x: x[0]) #based on keys
imax,vmax=max(Ncl.items(), key = lambda x: x[1])#based on values % find which class has most samples in K neighbors

# % Step 7: predict the class of the test sample to the majority class in K neighbors
Cpred = imax

print("Predicted Class: ",Cpred)