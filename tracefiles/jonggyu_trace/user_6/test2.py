import os
import numpy as np
import random
import re
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors, datasets
from sklearn import cluster, datasets, preprocessing
from sklearn.metrics import precision_recall_curve

user0 = [
14665,
14645,
14625,
15861
]

user1 = [
14892,
14866,
14828,
14802
]

user2 = [
15139,
15115,
15093,
15069
]

user3 = [
15445,
15423,
15401,
15377
]

user4 = [
14494,
14475,
14458
]

user5 = [
17277,
17249,
17158
]

def extract_features(): 
    final = []
    final_x = []
    final_y = []
    f = open ('c.c', 'r')
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
	elif int(values[0]) in user4:
            final_y.append(4)
	elif int(values[0]) in user5:
            final_y.append(5)

        else:
            final_x.pop()
        final = []
 
#    print final_y
    return final_x, final_y

def make_array (y):
#    x = y[:3000]
    x=y
    p = int(len(x)*0.1)
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
clf =neighbors.KNeighborsClassifier(6, 'uniform')
clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
score=0


for i in range(len(test_x)):
    if (test_y[i] == predict_y[i]):
        score += 1
print score
print "accuracy = ", float(float(score)/float(len(predict_y)))* 100

