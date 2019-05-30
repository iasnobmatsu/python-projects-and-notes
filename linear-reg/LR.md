# Linear Regression
import modules
```from sklearn import preprocessing, cross_validation, linear_model
import matplotlib.pyplot as plt
import pandas as pd
```

refine data
```
energy.head()
energy["Relative Compactness"]=energy["X1"]
energy["Surface Area"]=energy["X2"]
energy["Wall Area"]=energy["X3"]
energy["Roof Area"]=energy["X4"]
energy["Overall Height"]=energy["X5"]
energy["Glazing Area"]=energy["X7"]
energy["Orientation"]=energy["X6"]
energy["Glazing Area Distribution"]=energy['X8']
energy["Heating"]=energy["Y1"]
energy["Cooling"]=energy["Y2"]
energy=energy[["Relative Compactness","Surface Area","Wall Area","Roof Area","Overall Height","Orientation","Glazing Area","Glazing Area Distribution","Heating","Cooling"]]
energy.head()
```

visualization
```
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax1.set_xlabel("Surface Area")
ax1.set_ylabel("heating load")
ax1.scatter(energy["Surface Area"], energy["Heating"], color="red",s=1)
ax2=fig.add_subplot(1,2,2)
ax2.set_xlabel("Surface Area")
ax2.set_ylabel("cooling load")
ax2.scatter(energy["Surface Area"], energy["Cooling"], color="blue",s=1)
plt.show()
```

preprocessing, apply algorithm, testing
```
#preprocessing
X=np.array(energy[["Relative Compactness","Surface Area","Wall Area","Roof Area","Overall Height","Orientation","Glazing Area","Glazing Area Distribution"]])
y=np.array(energy["Heating"])
X=preprocessing.scale(X)
#cross validation
X_train, X_test, y_train,y_test=cross_validation.train_test_split(X, y, test_size=0.15)#0.15 is test sample size

clf=linear_model.LinearRegression()
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))
```

