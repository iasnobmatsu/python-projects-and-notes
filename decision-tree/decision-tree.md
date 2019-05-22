# Decision Tree
A Decision Tree is a tree-shaped diagram for making predictions or decisions. 

### CART: Classification and Regression Trees
classification: where the result of tree is discrete.

regression: where the result of tree is continuous.

### Splitting
splitting given data into a tree structure based on cost of splitting.

**Greedy splitting** calculates cost of splitting for all attributes in a dataset and chooses the one with minimun cost as next splitting node, repeat until node is data. 
There are many ways to split a tree.
- Gini
    - Gini Impurity: `IG=sum(p(i)*(1-p(i)))=1-sum(p(i)**2)` where p(i) is proportion of sample belong to an attribute at a particular node.
    - Gini Infomation Gain: Gini impurity at target (the decision)-weigted average of Gini impurity at different categories of an attribute. Calculated separately for each attribute.
- Entropy
    - Entropy:`IH=-sum(p(i)*log2(p(i))` where p(i) is proportion of sample belong to an attribute at a particular node.
    - Entropy Information Gain: Entropy at decision - weighted average of entropy at each category within an attribute. Calculated separately for each attribute.

### Pruning
After finding out information gain, the attribute with the next max information gain is the next splitting node. 
Pruning can be used to limite number of splits in a tree in case a data set creates too many splits.

### Python Implementation
import packages.
```
from sklearn import tree
from sklearn.datasets import load_iris
import graphviz
from sklearn.model_selection import train_test_split
```
load data and plot tree.
```
iris_data=load_iris()
iris_data.data#data for the 4 attributes
iris_data.feature_names # 4 attributes in columns
iris_data.target #labels, classifications-classes of iris

iris_tree=tree.DecisionTreeClassifier(max_depth=2)
iris_tree=iris_tree.fit(iris_data.data, iris_data.target)

# print (iris_data.DESCR)
```
predict result.
```
iris_tree.predict([[8,5,6,2]])#predict which class an iris is based on its features
iris_tree.predict_proba([[8,5,6,2]])#predict probability of the iris being in class 0, 1, and 2
```
split dataset into data and test sets.
```
data_train, data_test, target_train,target_test=train_test_split(iris_data.data, 
iris_data.target, test_size=0.2, random_state=None)
iris_tree=tree.DecisionTreeClassifier()
iris_tree.fit(data_train, target_train)
score=iris_tree.score(data_test, target_test)
print(score)
```

### Reference
[sklearn documentation](https://scikit-learn.org/stable/modules/tree.html)
[Deep Learning NLP documentation](https://machine-learning-course.readthedocs.io/en/latest/content/supervised/decisiontrees.html#cost-of-splitting)
[Classification And Regression Trees for Machine Learning](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)
[A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#three)
