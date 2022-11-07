#!/usr/bin/env python3

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import csv

reader = csv.reader(open("DATASET_Navigation_2022_2_edit.csv"))
reader.__next__()
X = []
Y1 = []
Y2 = []
for row in reader:
  row = list(map(float, row))
  X.append(row[:6])
  Y1.append(row[6])
  Y2.append(row[7])

X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y1, test_size=0.33)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, Y2, test_size=0.33)

dtc1 = DecisionTreeRegressor()
dtc1.fit(X1_train, y1_train)
print("dtc1 score", dtc1.score(X1_test, y1_test))

dtc2 = DecisionTreeRegressor()
dtc2.fit(X2_train, y2_train)
print("dtc2 score", dtc2.score(X2_test, y2_test))

rfc1 = RandomForestRegressor()
rfc1.fit(X1_train, y1_train)
print("rfc1 score", rfc1.score(X1_test, y1_test))

rfc2 = RandomForestRegressor()
rfc2.fit(X2_train, y2_train)
print("rfc2 score", rfc2.score(X2_test, y2_test))
