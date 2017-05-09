import time;
from gensim.models import Word2Vec
import pandas as pd
import time
from nltk.corpus import stopwords
import numpy as np
from KaggleWord2VecUtility import KaggleWord2VecUtility
from numpy import float32
import math
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn import svm
import pickle
import cPickle
from math import *
from sklearn.mixture import GMM

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cluster_GMM(num_clusters, word_vectors):
	# Initalize a GMM object and use it for clustering.
	clf =  GMM(n_components=num_clusters,
                    covariance_type="tied", init_params='wc', n_iter=10)
	# Get cluster assignments.
	idx = clf.fit_predict(word_vectors)
	print "Clustering Done...", time.time()-start, "seconds"
	# Get probabilities of cluster assignments.
	idx_proba = clf.predict_proba(word_vectors)
	# Dump cluster assignments and probability of cluster assignments. 
	joblib.dump(idx, 'gmm_latestclusmodel_len2alldata.pkl')
	print "Cluster Assignments Saved..."

	joblib.dump(idx_proba, 'gmm_prob_latestclusmodel_len2alldata.pkl')
	print "Probabilities of Cluster Assignments Saved..."
	return (idx, idx_proba)

def read_GMM(idx_name, idx_proba_name):
	# Loads cluster assignments and probability of cluster assignments. 
	idx = joblib.load(idx_name)
	idx_proba = joblib.load(idx_proba_name)
	print "Cluster Model Loaded..."
	return (idx, idx_proba)

def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict):
	# This function computes probability word-cluster vectors.
	prob_wordvecs_idf_len2alldata = {}

	i = 0
	for word in featurenames:
		i += 1
		if word in word_centroid_map:	
			prob_wordvecs_idf_len2alldata[word] = {}
			for index in range(0, num_clusters):
					prob_wordvecs_idf_len2alldata[word][index] = model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word] 

	prob_wordvecs = {}

	for word in prob_wordvecs_idf_len2alldata.keys():
		prob_wordvecs[word] = prob_wordvecs_idf_len2alldata[word][0]
		for index in prob_wordvecs_idf_len2alldata[word].keys():
			if index==0:
				continue
			prob_wordvecs[word] = np.concatenate((prob_wordvecs[word], prob_wordvecs_idf_len2alldata[word][index]), axis=1)
	
	return prob_wordvecs

def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, word_centroid_prob_map, dimension, word_idf_dict, featurenames, num_centroids, train=False):
	# This function computes SDV feature vectors.
	bag_of_centroids = np.zeros( num_centroids * dimension, dtype="float32" )
	global min_no
	global max_no

	for word in wordlist:
		try:
			temp = word_centroid_map[word]
		except:
			continue

		bag_of_centroids += prob_wordvecs[word]

	norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
	if(norm!=0):
		bag_of_centroids /= norm

	# To make feature vector sparse, make note of minimum and maximum values.
	if train:
		min_no += min(bag_of_centroids)
		max_no += max(bag_of_centroids)

	return bag_of_centroids

if __name__ == '__main__':

	start = time.time()

	num_features = int(sys.argv[1])     # Word vector dimensionality
	min_word_count = 20   # Minimum word count
	num_workers = 40       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words

	model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context_len2alldata"
  	# Load the trained Word2Vec model.
  	model = Word2Vec.load(model_name)
  	# Get wordvectors for all words in vocabulary.
	word_vectors = model.syn0

	# Load train data.
	train = pd.read_csv( 'data/train_v2.tsv', header=0, delimiter="\t")
	# Load test data.
	test = pd.read_csv( 'data/test_v2.tsv', header=0, delimiter="\t")
	all = pd.read_csv( 'data/all_v2.tsv', header=0, delimiter="\t")

	# Set number of clusters.
	num_clusters = int(sys.argv[2])
	# Uncomment below line for creating new clusters.
	idx, idx_proba = cluster_GMM(num_clusters, word_vectors)

	# Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
	#idx_name = "gmm_latestclusmodel_len2alldata.pkl"
	#idx_proba_name = "gmm_prob_latestclusmodel_len2alldata.pkl"
	#idx, idx_proba = read_GMM(idx_name, idx_proba_name)

	# Create a Word / Index dictionary, mapping each vocabulary word to
	# a cluster number
	word_centroid_map = dict(zip( model.index2word, idx ))
	# Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
	# list of probabilities of cluster assignments.
	word_centroid_prob_map = dict(zip( model.index2word, idx_proba ))

	# Computing tf-idf values.
	traindata = []
	for i in range( 0, len(all["news"])):
		traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["news"][i], True)))

	tfv = TfidfVectorizer(strip_accents='unicode',dtype=np.float32)
	tfidfmatrix_traindata = tfv.fit_transform(traindata)
	featurenames = tfv.get_feature_names()
	idf = tfv._tfidf.idf_

	# Creating a dictionary with word mapped to its idf value 
	print "Creating word-idf dictionary for Training set..."

	word_idf_dict = {}
	for pair in zip(featurenames, idf):
		word_idf_dict[pair[0]] = pair[1]

	# Pre-computing probability word-cluster vectors.
	prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)

	temp_time = time.time() - start
	print "Creating Document Vectors...:", temp_time, "seconds."

	# gwbowv is a matrix which contains normalised document vectors.
	gwbowv = np.zeros( (train["news"].size, num_clusters*(num_features)), dtype="float32")

	counter = 0

	min_no = 0
	max_no = 0
	for review in train["news"]:
		# Get the wordlist in each news article.
		words = KaggleWord2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True )
		gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters, train=True)
		counter+=1
		if counter % 1000 == 0:
			print "Train News Covered : ",counter

 	gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"



	gwbowv_test = np.zeros( (test["news"].size, num_clusters*(num_features)), dtype="float32")

	counter = 0

	for review in test["news"]:
		# Get the wordlist in each news article.
		words = KaggleWord2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True )
		gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters)
		counter+=1
		if counter % 1000 == 0:
			print "Test News Covered : ",counter

    
	test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

	print "Making sparse..."
	# Set the threshold percentage for making it sparse. 
	percentage = 0.04
	min_no = min_no*1.0/len(train["news"])
	max_no = max_no*1.0/len(train["news"])
	print "Average min: ", min_no
	print "Average max: ", max_no
	thres = (abs(max_no) + abs(min_no))/2
	thres = thres*percentage

	# Make values of matrices which are less than threshold to zero.
	temp = abs(gwbowv) < thres
	gwbowv[temp] = 0

	temp = abs(gwbowv_test) < thres
	gwbowv_test[temp] = 0
	
	#saving gwbowv train and test matrices
	np.save(gwbowv_name, gwbowv)
	np.save(test_gwbowv_name, gwbowv_test)
	
	endtime = time.time() - start
	print "SDV created and dumped: ", endtime, "seconds."
	print "Fitting a SVM classifier on labeled training data..."
	print "5 fold cross validation on multiple parameters"

	param_grid = [
	  {'C': np.arange(0.1, 10, 0.5)}
	 ]
	scores = ['accuracy', 'f1_micro' , 'f1_macro' , 'f1_weighted' ,  'recall_micro',  'precision_micro', 'recall_macro',  'precision_macro', 'recall_weighted', 'precision_weighted']
	for score in scores:
		print "# Tuning hyper-parameters for", score, "\n"
		clf = GridSearchCV(LinearSVC(C=1), param_grid, cv=5, scoring= '%s' % score)
		clf.fit(gwbowv, train["class"])
		print "Best parameters set found on development set:\n"
		print clf.best_params_
		print "Best value of ", score, ":\n"
		print clf.best_score_
		Y_true, Y_pred  = test["class"], clf.predict(gwbowv_test,test["class"])
		print "Accuracy for score ", score, " :",clf.score(gwbowv_test,test["class"])
		print classification_report(Y_true, Y_pred)
	
	endtime = time.time()
	print "Total time taken: ", endtime-start, "seconds." 

	print "********************************************************"
