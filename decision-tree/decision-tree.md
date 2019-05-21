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


### Reference
[Deep Learning NLP documentation](https://machine-learning-course.readthedocs.io/en/latest/content/supervised/decisiontrees.html#cost-of-splitting)

[Classification And Regression Trees for Machine Learning](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)

[A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/#three)