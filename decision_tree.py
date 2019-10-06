from sklearn import tree
from sklearn.model_selection import KFold
import numpy as np
N = 67557
inp = [[0 for i in range(0 , 42)] for i in range(0 , N)]
val = [0] * N
for i in range(0 , N):
	x = input().split(",")
	inp[i] = x[0:42]
	if x[42] == "win":
		val[i] = 0
	elif x[42] == "loss":
		val[i] = 1
	else:
		val[i] = 2
	for j in range(0 , 42):
		if x[j] == 'b':
			inp[i][j] = 0
		elif x[j] == 'x':
			inp[i][j] = 1
		else:
			inp[i][j] = 2
inp = np.array(inp)
val = np.array(val)
abcd = tree.DecisionTreeClassifier()
kf = KFold(n_splits = 10)
kf.get_n_splits(inp)
for trainindex , testindex in kf.split(inp):
	X_train , X_test = inp[trainindex] , inp[testindex]
	Y_train , Y_test = val[trainindex] , val[testindex]
	abcd.fit(X_train , Y_train)
	vals = abcd.predict(X_test)
	correct = 0
	wrong = 0
	for j in range(0 , len(vals)):
		if vals[j] == val[testindex[j]]:
			correct += 1
		else:
			wrong += 1
	print ("accuracy is " + str((1.0 * correct / (correct + wrong)) * 100) + "%")
