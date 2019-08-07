import os
import numpy as np
import random
import re
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors, datasets
from sklearn import cluster, datasets, preprocessing
from sklearn.metrics import precision_recall_curve

user0 = [13091,
12638,
13868,
17830,
17813,
20349,
20169,
19916,
24218,
24006,
23885,
23799,
25891,
26115,
26183]

user1 = [
14186,
14279,
14339,
17407,
18328,
21116,
21345,
21485,
24345,
23568,
24651,
26452,
26474,
26497,
26519,
]

user2 = [
14518,
14546,
14572,
18891,
21667,
21716,
21757,
24865,
25041,
24919,
27678,
27705,
27731,
]

user3 = [
15709,
15594,
14787,
18185,
21891,
21913,
21929,
25677,
25709,
25728,
25742,
]


def extract_features(): 
    final = []
    final_x = []
    final_y = []
    f = open ('nemo.log', 'r')
    content2 = f.read().split('\n')
    content = filter(None, content2)
    random.shuffle(content)

    for s in range(len(content)-1) :
        values = re.findall(r'[-+]?\d*\.\d+|\d+',''.join(content[s-1]))
        final.append(float(values[1]))
        final.append(float(values[2]))
        final.append(float(values[3]))
        final.append(float(values[4]))
        final.append(float(values[5]))
        final_x.append(final)
        if int(values[0]) in user0:
            final_y.append(0)
        elif int(values[0]) in user1:
            final_y.append(1)
        elif int(values[0]) in user2:
            final_y.append(2)
        elif int(values[0]) in user3:
            final_y.append(3)
        else:
            final_x.pop()
        final = []
 
#    print final_y
    return final_x, final_y

def make_array (x):
    p = int(len(x)*0.01)
#    print "total = ", len(x), "  p = ", p
    a_x = x[:p]
    b_x = x[p:]
    return a_x, b_x



x_array, y_array = extract_features()
train_x, test_x = make_array(x_array)
train_y, test_y = make_array(y_array)


#nbrs =neighbors.KNeighborsClassifier(4, algorithm='ball_tree').fit(train_x)
#distances, indices = nbrs.kneighbors(X)
#print distances


#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)


# K-means
#k_means = cluster.KMeans(n_clusters=4)
#k_means.fit(train_x)
#print(list11)
#print(train_y)


# K-NN
clf =neighbors.KNeighborsClassifier(4, 'uniform')
clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
score=0


for i in range(len(test_x)):
    if (test_y[i] == predict_y[i]):
        score += 1
    else:
        print predict_y[i]
print score
print "accuracy = ", float(float(score)/float(len(predict_y)))* 100

