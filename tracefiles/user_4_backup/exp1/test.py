from __future__ import division
import collections
import random 
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score

def number_of_chars(s):
    return len(s)

def unique_chars(s):
    s2 = ''.join(set(s))
    return len(s2)

def weighted_unique_chars(s):
    return unique_chars(s)/number_of_chars(s)

def words_count(s):
    return collections.Counter(s)

def words_counter_object(s):
    cnt = collections.Counter()

    words = s.split()	
    for w in words:
        cnt[w] += 1

    return cnt

def total_words(cnt):
    sum = 0
    for k in dict(cnt).keys():
        sum += int(cnt[k])

    return sum

def most_common(cnt, n):
    for k,v in cnt.most_common(n):
        #print "most common  k = %s : v = %s" %(k,v)
        pass

def is_repeated(cnt):
    for k,v in cnt.most_common(1):
        freq = v/total_words(cnt)
        # print 'freq=',freq
        if freq > 0.5:
            return 1
    return 0

def make_feature_vector(critique, labels):
    " construct feature vector" 
    feature_vector = []
    for i in range(len(critique)):
        s = critique[i]
        feature = []
        counter_obj = words_counter_object(s)

        feature.append(number_of_chars(s))
        feature.append(unique_chars(s))
        feature.append(weighted_unique_chars(s))
        feature.append(total_words(counter_obj))
        feature.append(is_repeated(counter_obj))

        feature.append(labels[i])
        feature_vector.append(feature)

    return feature_vector

def read_data():
    ''' read and make a list of critiques'''
    f = open('bad.txt', 'r')
    bad = f.read().split('|')
    f.close()

    f = open('good.txt', 'r')
    good = f.read().split('|')
    f.close()

    return bad+good, [0]*len(bad) + [1]*len(good)

def make_np_array_XY(xy):
    print "make_np_array_XY()"
    a = np.array(xy)
    x = a[:,0:-1]
    y = a[:,-1]
    return x,y

if __name__ == '__main__':

    critiques, labels = read_data()
    features_and_labels= make_feature_vector(critiques, labels)
    number_of_features = len(features_and_labels[0]) - 1
    random.shuffle(features_and_labels)

    # make  train / test sets from the shuffled list
    cut = int(len(features_and_labels)/2)
    XY_train = features_and_labels[:cut]
    XY_test = features_and_labels[cut:]

    X_train, Y_train = make_np_array_XY(XY_train)

    X_test, Y_test = make_np_array_XY(XY_test)

	# train set
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
    print 'type(svc)=', type(svc)
    print 'svc=',svc

    print 'Y_test:\n', Y_test

    Y_predict = svc.predict(X_test)
    print 'Y_predict:\n', Y_predict

    # score
    test_size = len(Y_test)
    score = 0
    for i in range(test_size):
        if Y_predict[i] == Y_test[i]:
            score += 1

    print 'Got %s out of %s' %(score, test_size)
  
    # f1 score
    f1 = f1_score(Y_test, Y_predict, average='macro')  
    print 'f1 macro = %.2f' %(f1)
    f1 = f1_score(Y_test, Y_predict, average='micro')  
    print 'f1 micro = %.2f' %(f1)
    f1 = f1_score(Y_test, Y_predict, average='weighted')  
    print 'f1 weighted = %.2f' %(f1)
