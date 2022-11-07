#!/usr/bin/env python3

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import csv

reader = csv.reader(open("DATASET_Navigation_2022_2_edit.csv"))
reader.__next__()
X = []
Y1 = []
Y = []
Y2 = []
for row in reader:
  row = list(map(float, row))
  X.append(row[:6])
  Y.append([row[6], row[7]])
  Y1.append(row[6])
  Y2.append(row[7])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y1, test_size=0.33)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, Y2, test_size=0.33)

dtc1 = DecisionTreeRegressor()
dtc1.fit(X1_train, y1_train)
print("dtc1 score", dtc1.score(X1_test, y1_test))

dtc2 = DecisionTreeRegressor()
dtc2.fit(X2_train, y2_train)
print("dtc2 score", dtc2.score(X2_test, y2_test))
print("dtc12 avg score", (dtc1.score(X1_test, y1_test) + dtc2.score(X2_test, y2_test))/2)

for msl in [10, 3, 1]:
  for depth in [2, 9, 13, None]:
    dtc = DecisionTreeRegressor(max_depth = depth, min_samples_leaf=msl)
    dtc.fit(X_train, y_train)
    if depth==None:
      depth = dtc.get_depth()
    print("%f"%dtc.score(X_test, y_test), "&", depth, "&", msl, "\\\\")

#import graphviz 
#dot_data = tree.export_graphviz(dtc, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris") 



rfc1 = RandomForestRegressor()
rfc1.fit(X1_train, y1_train)

rfc2 = RandomForestRegressor()
rfc2.fit(X2_train, y2_train)
print("rfc1 score", rfc1.score(X1_test, y1_test))
print("rfc2 score", rfc2.score(X2_test, y2_test))
print("rfc12 avg score", (rfc1.score(X1_test, y1_test) + rfc2.score(X2_test, y2_test))/2)

for msl in [10, 5, 3, 1]:
  for depth in [2, 8, 15, None]:
    rfc = RandomForestRegressor(max_depth = depth, min_samples_leaf=msl)
    rfc.fit(X_train, y_train)
    print("%f"%rfc.score(X_test, y_test), "&", depth, "&", msl, "\\\\")

