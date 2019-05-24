# K Nearest Neighbours
KNN is a classifying algorithm that makes decisions based on k selected items that have distance closest to the target item.
### Distance
Euclidean Distance could be used for finding nearest neighbours.
```
d = sqrt((x1-x2)**2+(y1-y2)**2+...+(n1-n2)**2)
for points (x1,y1,...,n1) and (x2,y2,...n2) 
```
### KD Tree and Ball Tree
These are algorithms used to speed up the process of finding the distance between each item. The working principle involves separating nearby data points into one branch and calculated nearby distance first to save time. 

### Python Implementation
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import NearestNeighbors
import matplotlib as plt
```
cancer_data=load_breast_cancer().data[:,:10]
cancer_target=load_breast_cancer().target
print(cancer.DESCR)
knn=KNeighborsClassifier(n_neighbors=10,algorithm="auto")
# use a small much not too small n_neighbout to exclude outliers and fit the model
knn.fit(cancer_data, cancer_target)
knn.predict([[6.9,9,40,140,0.005,0.01,0.1,0.1,0.1,0.1]])
```