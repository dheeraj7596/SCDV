import time
import warnings
import os
import sys
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import multiprocessing
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from gensim.corpora.wikicorpus import WikiCorpus
import pandas as pd
import numpy as np
from numpy import float32
import math
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
import pickle
import cPickle
from math import *
from sklearn.mixture import GMM
from gensim.models import TfidfModel


def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def cluster_GMM(num_clusters, word_vectors):
	# Initalize a GMM object and use it for clustering.
	clf =  GMM(n_components=num_clusters,
                    covariance_type="tied", init_params='wc', n_iter=100)
	# Get cluster assignments.
	idx = clf.fit_predict(word_vectors)
	# Get probabilities of cluster assignments.
	idx_proba = clf.predict_proba(word_vectors)
	# Dump cluster assignments and probability of cluster assignments. 
	joblib.dump(idx, folder + '/' + 'gmm_latestclusmodel_len2alldata.pkl')
	print "Cluster Assignments Saved..."

	joblib.dump(idx_proba, folder + '/' + 'gmm_prob_latestclusmodel_len2alldata.pkl')
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
	global word_centroid_prob_map
	global model

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
		# print word
		prob_wordvecs[word] = prob_wordvecs_idf_len2alldata[word][0]
		for index in prob_wordvecs_idf_len2alldata[word].keys():
			# print index
			if index==0:
				continue
			prob_wordvecs[word] = np.concatenate((prob_wordvecs[word], prob_wordvecs_idf_len2alldata[word][index]))
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
	# print bag_of_centroids
	return bag_of_centroids


def create_document_vectors(folder, clustering, make_idf, make_wordtopic_vectors):

	global min_no
	global max_no
	global word_centroid_prob_map
	global model

	start = time.time()
	cores = multiprocessing.cpu_count()
	num_features = int(sys.argv[1])     # Word vector dimensionality
	min_word_count = 3   # Minimum word count
	num_workers = max(1, cores - 1)     # Number of threads to run in parallel
	context = 5          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words
	document_list = os.listdir(folder + '/' + "raw")


	# load skip gram pretrained word vectors 
	word2vec_output_file = folder + '/' + "word2vec.model"
	model = Word2Vec.load(word2vec_output_file)
	word_vectors = model.syn0

	print "Finished loading word vectors"

	num_clusters = int(sys.argv[2])


	# cluster word embeddings using GMM
	if clustering  == True:
		idx, idx_proba = cluster_GMM(num_clusters, word_vectors)
	else:
		idx_name = folder + '/' + "gmm_latestclusmodel_len2alldata.pkl"
		idx_proba_name = folder + '/' + "gmm_prob_latestclusmodel_len2alldata.pkl"
		idx, idx_proba = read_GMM(idx_name, idx_proba_name)

	word_centroid_map = dict(zip( model.index2word, idx ))
	word_centroid_prob_map = dict(zip( model.index2word, idx_proba ))

	# compute idf values for the corpus
	if make_idf == True:
		data = []
		for filename in document_list:
			data.append(open(folder + '/' + "raw" + "/" + filename).read())
		#Code to create a idf and featuename vectors from corpus
		tfv = TfidfVectorizer(strip_accents=None, dtype=np.float32, vocabulary=model.index2word)
		tfidfmatrix_traindata = tfv.fit_transform(data)
		featurenames = tfv.get_feature_names()
		idf = tfv._tfidf.idf_
		np.save(folder + '/' + "featurenames.npy", featurenames)
		np.save(folder + '/' + "idf.npy", idf)
	else:
		featurenames = np.load(folder + '/' + "featurenames.npy")
		idf = np.load(folder + '/' + "idf.npy")

	print "unique tokens recognized by Word2Vec:" + str(len(word_centroid_map))
	print "unique tokens recognized by TfidfVectorizer:" + str(len(featurenames))

	print "Creating word-idf dictionary for document set..."
	word_idf_dict = {}
	for pair in zip(featurenames, idf):
		word_idf_dict[pair[0]] = pair[1]	

	# compute SCDV vectors for terms in Vocabulary
	if make_wordtopic_vectors == True:
		print "Creating word-topic vectors"
		prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict)
		joblib.dump(prob_wordvecs, folder + '/' + "prob_wordvecs.pkl")
	else:
		print "Loading word-topic vectors"
		prob_wordvecs = joblib.load(folder + '/' + "prob_wordvecs.pkl")

	print "length of prob_wordvecs: " + str(len(prob_wordvecs))


	temp_time = time.time() - start

	# create SCDV vectors for documents 
	print "Creating Document Vectors...:", temp_time, "seconds."
	gwbowv = np.zeros( (len(document_list), num_clusters*(num_features)), dtype="float32")

	counter = 0
	min_no = 0
	max_no = 0
	docid_dictionary = {}
	for filename in document_list:
		# Get the wordlist in each news article.
		words = open(folder + '/' + "raw" + "/" + filename).read().split()
		gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters, train=True)
		docid_dictionary[counter] = filename.split('.')[0]
		counter+=1
		if counter % 100 == 0:
			print "Documents Covered : ",counter

 	gwbowv_name = folder + '/' + "DOC_SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"
 	joblib.dump(docid_dictionary, folder + '/' + 'docid_dictionary.pkl')

	print "Making sparse..."
	# Set the threshold percentage for making it sparse. 
	percentage = 0.04
	min_no = min_no*1.0/len(document_list)
	max_no = max_no*1.0/len(document_list)
	print "Average min: ", min_no
	print "Average max: ", max_no
	thres = (abs(max_no) + abs(min_no))/2
	thres = thres*percentage
	print "throshold:" + str(thres)
	# Make values of matrices which are less than threshold to zero.
	temp = abs(gwbowv) < thres
	gwbowv[temp] = 0

	# np.save(gwbowv_name, gwbowv)

 	# create SCDV vectors for queries
 	query_list = {}
 	queryid_dictionary = {}
 	query_file = open(folder + '/' + "queries.txt").readlines()
 	i = 0
 	while i < len(query_file):
 		print query_file[i+1]
 		query_list[query_file[i]] = query_file[i+1]
 		i = i+2
 	counter = 0
 	gwbowv_query = np.zeros( (len(query_list), num_clusters*(num_features)), dtype="float32")	
 	for key, val in query_list.iteritems():
 		words = val.split()
 		gwbowv_query[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, word_centroid_map, word_centroid_prob_map, num_features, word_idf_dict, featurenames, num_clusters, train=False)
 		queryid_dictionary[counter] = key
 		counter+=1
		if counter % 10 == 0:
			print "Queries Covered : ",counter
	joblib.dump(queryid_dictionary, folder + '/' + 'queryid_dictionary.pkl')
	query_gwbowv_name = folder + '/' + "QUERY_SDV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_gmm_sparse.npy"

	temp = abs(gwbowv_query) < thres
	gwbowv_query[temp] = 0

	# np.save(query_gwbowv_name, gwbowv_query)

	# compute query-document score
	print gwbowv.shape
	print gwbowv_query.shape
	scores =  np.dot(gwbowv_query, np.transpose(gwbowv))
	docid_dictionary = joblib.load(folder + '/' + "docid_dictionary.pkl")
	queryid_dictionary = joblib.load(folder + '/' + "queryid_dictionary.pkl")
	result_file = folder + '/' + "result.pkl"
	joblib.dump(scores, result_file)
	counter = 0
	#save scores in a format that the IR system can consume
	f1 = open(folder + '/' + "SDVresult.txt", "w+")
	f2 = open(folder + '/' + "documentlist.txt", "w+")
	for query in scores:
		print query
		f1.write(queryid_dictionary[counter].strip() + " ")
		f1.write(" ".join([str(x) for x in query]))
		f1.write("\n")
		counter += 1
	document_list = os.listdir(folder + '/' + "raw")
	for c in range(0, len(document_list)):
		f2.write(docid_dictionary[c] + "\n")
	f1.close()
	f2.close()
	print "Task complete"

	endtime = time.time() - start
	print "Query-Document scores completed: ", endtime, "seconds."


if __name__ == '__main__':
	folder = sys.argv[3]
	# make_clusters = True if GMM needs to be run for the first time (one time operation)
	make_clusters = True
	# make_idf = True if idf needs to be computed (one time operation)
	make_idf = True
	# make_wordtopic_vectors = True if term SCDV vectors need to be generated (one time operation)
	make_wordtopic_vectors = True

	create_document_vectors(folder, make_clusters, make_idf, make_wordtopic_vectors)

