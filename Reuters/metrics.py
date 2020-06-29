import time
import pandas as pd
import numpy as np
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


print("Fitting One vs Rest SVM classifier to labeled cluster training data...")

start = time.time()

num_features = int(sys.argv[1])
num_clusters = int(sys.argv[2])

all = pd.read_pickle('all.pkl')

# Get train and text data.
lb = MultiLabelBinarizer()
Y = lb.fit_transform(all.tags)
train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

train = DataFrame({'text': []})
test = DataFrame({'text': []})

train["text"] = train_data.reset_index(drop=True)
test["text"] = test_data.reset_index(drop=True)

# Load feature vectors.
gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"
gwbowv = np.load(gwbowv_name)

test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"
gwbowv_test = np.load(test_gwbowv_name)

param_grid = [
    {'estimator__C': np.arange(0.1, 150, 10)}
    # {'C': [1, 10, 100, 200, 500, 1000], 'gamma': [0.01, 0.05, 0.001, 0.005,  0.0001], 'kernel': ['rbf']},
]
scores = ['f1_weighted']  # , 'accuracy', 'recall', 'f1']
for score in scores:
    strt = time.time()
    print("# Tuning hyper-parameters for", score, "\n")
    clf = GridSearchCV(OneVsRestClassifier(LogisticRegression(C = 100.0), n_jobs=30), param_grid, cv=5, n_jobs=30, scoring= '%s' % score)
    clf = clf.fit(gwbowv, Y_train)

    pred = clf.predict(gwbowv_test)
    pred_proba = clf.predict_proba(gwbowv_test)

    K = [1,3,5]

    for k in K:
        Total_Precision = 0
        Total_DCG = 0
        norm = 0
        for i in range(k):
            norm += 1 / math.log(i + 2)

        loop_var = 0
        for item in pred_proba:
            classelements = sorted(range(len(item)), key=lambda i: item[i])[-k:]
            classelements.reverse()
            precision = 0
            dcg = 0
            loop_var2 = 0
            for element in classelements:
                if Y_test[loop_var][element] == 1:
                    precision += 1
                    dcg += 1 / math.log(loop_var2 + 2)
                loop_var2 += 1

            Total_Precision += precision * 1.0 / k
            Total_DCG += dcg * 1.0 / norm
            loop_var += 1
        print("Precision@", k, ": ", Total_Precision * 1.0 / loop_var)
        print("NDCG@", k, ": ", Total_DCG * 1.0 / loop_var)

    print("Coverage Error: ", coverage_error(Y_test, pred_proba))
    print("Label Ranking Average precision score: ", label_ranking_average_precision_score(Y_test, pred_proba))
    print("Label Ranking Loss: ", label_ranking_loss(Y_test, pred_proba))
    print("Hamming Loss: ", hamming_loss(Y_test, pred))
    print("Weighted F1score: ", f1_score(Y_test, pred, average='weighted'))

    # print "Total time taken: ", time.time()-start, "seconds."

endtime = time.time()
print("Total time taken: ", endtime - start, "seconds.")

print("********************************************************")
