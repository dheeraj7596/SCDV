import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec, FastText
from KaggleWord2VecUtility import KaggleWord2VecUtility
import pandas as pd
import time
import numpy as np
import sys
import joblib
from sklearn.mixture import GaussianMixture as GMM
import collections
import math


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def cluster_GMM(num_clusters, word_vectors):
    # Initalize a GMM object and use it for clustering.
    clf = GMM(n_components=num_clusters, covariance_type="tied", init_params='wc', n_iter=10)
    # Get cluster assignments.
    idx = clf.fit_predict(word_vectors)
    print("Clustering Done...", time.time() - start, "seconds")
    # Get probabilities of cluster assignments.
    idx_proba = clf.predict_proba(word_vectors)
    # Dump cluster assignments and probability of cluster assignments.
    joblib.dump(idx, 'gmm_latestclusmodel_len2alldata_80.pkl')
    print("Cluster Assignments Saved...")
    joblib.dump(idx_proba, 'gmm_prob_latestclusmodel_len2alldata_80.pkl')
    print("Probabilities of Cluster Assignments Saved...")
    return (idx, idx_proba)


def read_GMM(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments.
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print("Cluster Model Loaded...")
    return (idx, idx_proba)


def get_probability_words(word_centroid_map, traindata):
    print("Getting probability words ...")
    prob_word = {}
    total_count = 0
    for item in traindata:
        for word in item.split():
            try:
                a = word_centroid_map[word]
                try:
                    prob_word[word] += 1
                    total_count += 1
                except:
                    prob_word[word] = 1
                    total_count += 1
            except:
                a = 0
    prob_word = {k: (v * 1.0) / (total_count) for k, v in (prob_word).iteritems()}
    return prob_word


def get_doccofrequency(traindata):
    print("Getting Doc-frequency and co-frequency")
    doc_cofreq = {}
    doc_freq = {}
    for item in traindata:
        unique_item = list(set(item.split()))
        for row in unique_item:
            try:
                doc_freq[row] += 1
            except:
                doc_cofreq[row] = {}
                doc_freq[row] = 1
            for col in unique_item:
                if row != col:
                    try:
                        doc_cofreq[row][col] += 1
                    except:
                        doc_cofreq[row][col] = 1
    return (doc_freq, doc_cofreq)


def get_probability_topic_vectors(word_centroid_map, num_clusters, prob_word):
    topic_centroid_prob_map = {}
    prob_topic = {}
    for index in range(0, num_clusters):
        print("Cluster-", index)
        topic_centroid_prob_map[index] = {}
        prob_topic[index] = 0
        print("Entering inner for loop...")
        for word in word_centroid_map:
            try:
                topic_centroid_prob_map[index][word] = word_centroid_prob_map[word][index] * prob_word[word]
                prob_topic[index] += word_centroid_prob_map[word][index] * prob_word[word]
            except:
                continue
        print("Exiting inner for loop...")
        topic_centroid_prob_map[index] = {k: (v * 1.0) / (prob_topic[index]) for k, v in
                                          (topic_centroid_prob_map[index]).iteritems()}
    return (prob_topic, topic_centroid_prob_map)


def get_coherence(doc_cofreq, doc_freq, num_clusters, num_topwords):
    print("Getting Coherence...")
    topic_coherence = {}
    overall_coherence = 0
    top10words = {}
    for index in range(0, num_clusters):
        most_probs = collections.Counter(topic_centroid_prob_map[index])
        require_mostcommons = most_probs.most_common(num_topwords)
        top10words[index] = require_mostcommons
        for i in range(1, num_topwords):
            for j in range(0, i):
                try:
                    temp = doc_cofreq[require_mostcommons[i][0]][require_mostcommons[j][0]]
                except:
                    temp = 0
                a = math.log(((temp + 1) * 1.0) / (doc_freq[require_mostcommons[j][0]]))
                try:
                    topic_coherence[index] += a
                except:
                    topic_coherence[index] = a
        overall_coherence += topic_coherence[index]
    return (topic_coherence, overall_coherence / num_clusters, top10words)


def get_pmi(doc_cofreq, doc_freq, num_clusters, num_topwords):
    print("Getting PMI...")
    topic_coherence = {}
    overall_coherence = 0
    top10words = {}
    for index in range(0, num_clusters):
        most_probs = collections.Counter(topic_centroid_prob_map[index])
        require_mostcommons = most_probs.most_common(num_topwords)
        top10words[index] = require_mostcommons
        for i in range(1, num_topwords):
            for j in range(0, i):
                try:
                    temp = doc_cofreq[require_mostcommons[i][0]][require_mostcommons[j][0]]
                except:
                    temp = 0
                a = math.log(
                    ((temp + 1) * 1.0) / (doc_freq[require_mostcommons[i][0]] * doc_freq[require_mostcommons[i][0]]))
                try:
                    topic_coherence[index] += a
                except:
                    topic_coherence[index] = a
        overall_coherence += topic_coherence[index]
    return (topic_coherence, overall_coherence / num_clusters, top10words)


def get_topic_document_prob_map(traindata, word_centroid_prob_map, num_clusters):
    topic_document_probability = {}
    for doc_index in len(traindata):
        overall_value = 0
        topic_document_probability[doc_index] = {}
        for index in range(0, num_clusters):
            topic_document_probability[doc_index][index] = 0
            for word in traindata[doc_index]:
                topic_document_probability[doc_index][index] += word_centroid_prob_map[word][index]
            overall_value += topic_document_probability[doc_index][index]
        topic_document_probability[doc_index] = {k: (v * 1.0) / (overall_value) for k, v in
                                                 (topic_document_probability[doc_index]).iteritems()}
    return topic_document_probability


def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, word_centroid_prob_map, dimension,
                                     word_idf_dict, featurenames, num_centroids, train=False):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
    global min_no
    global max_no

    for word in wordlist:
        try:
            temp = word_centroid_map[word]
        except:
            continue

        bag_of_centroids += prob_wordvecs[word]

    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if (norm != 0):
        bag_of_centroids /= norm

    # To make feature vector sparse, make note of minimum and maximum values.
    if train:
        min_no += min(bag_of_centroids)
        max_no += max(bag_of_centroids)

    return bag_of_centroids


if __name__ == '__main__':

    start = time.time()

    num_features = int(sys.argv[1])  # Word vector dimensionality
    min_word_count = 20  # Minimum word count
    num_workers = 40  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(
        context) + "context_len2alldata"

    model_type = sys.argv[4]
    assert model_type in ["word2vec", "fasttext"]

    if model_type == "word2vec":
        # Load the trained Word2Vec model.
        model = Word2Vec.load(model_name)
        # Get wordvectors for all words in vocabulary.
        word_vectors = model.wv.vectors
        index2word = model.wv.index2word
    elif model_type == "fasttext":
        # Load the trained FastText model.
        model = FastText.load(model_name)
        # Get wordvectors for all words in vocabulary.
        word_vectors = model.wv.vectors
        index2word = model.wv.index2word

    # Load train data.
    train = pd.read_csv('data/train_v2.tsv', header=0, delimiter="\t")
    # Load test data.
    test = pd.read_csv('data/test_v2.tsv', header=0, delimiter="\t")
    all = pd.read_csv('data/all_v2.tsv', header=0, delimiter="\t")

    # Set number of clusters.
    num_clusters = int(sys.argv[2])
    # Uncomment below line for creating new clusters.
    # idx, idx_proba = cluster_GMM(num_clusters, word_vectors)

    # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
    idx_name = "gmm_latestclusmodel_len2alldata_80.pkl"
    idx_proba_name = "gmm_prob_latestclusmodel_len2alldata_80.pkl"
    idx, idx_proba = read_GMM(idx_name, idx_proba_name)

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a index number
    word_centroid_map = dict(zip(index2word, idx))
    # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
    # list of probabilities of cluster assignments.
    word_centroid_prob_map = dict(zip(index2word, idx_proba))

    # Computing tf-idf values.
    traindata = []
    for i in range(0, len(all["news"])):
        traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(all["news"][i], True)))

    num_topwords = int(sys.argv[3])

    prob_word = get_probability_words(word_centroid_map, traindata)
    doc_freq, doc_cofreq = get_doccofrequency(traindata)
    prob_topic, topic_centroid_prob_map = get_probability_topic_vectors(word_centroid_map, num_clusters, prob_word)
    topic_coherence, overall_coherence, top10words = get_coherence(doc_cofreq, doc_freq, num_clusters, num_topwords)
    topic_pmi, overall_pmi, top10words_pmi = get_pmi(doc_cofreq, doc_freq, num_clusters, num_topwords)

    outfile = open("coherence_j.txt", "w")
    outfile.write(str(overall_coherence))
    outfile.write("\n")
    for i in range(num_clusters):
        for item in top10words[i]:
            outfile.write(str(item))
            outfile.write("\n")
        outfile.write(str(topic_coherence[i]))
        outfile.write("\n")
        outfile.write("**********************************************************")
        outfile.write("\n")
    outfile.close()

    outfile = open("pmi.txt", "w")
    outfile.write(str(overall_pmi))
    outfile.write("\n")
    for i in range(num_clusters):
        for item in top10words_pmi[i]:
            outfile.write(str(item))
            outfile.write("\n")
        outfile.write(str(topic_pmi[i]))
        outfile.write("\n")
        outfile.write("**********************************************************")
        outfile.write("\n")
    outfile.close()
