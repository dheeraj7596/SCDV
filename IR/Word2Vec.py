import multiprocessing
import sys
import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords

# logging is important to get the state of the functions
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    folder = sys.argv[2]
    document_list = os.listdir(folder + '/' + "raw")
    sentences = []
    vocab = []
    stops = set(stopwords.words("english"))
            
    for filename in document_list:
        sentences.append(open(folder + '/' + "raw" + "/" + filename).read().split())
    # sentences = open("sjm/sjm.txt").read().split()
    print len(sentences)
    params = {'size': int(sys.argv[1]), 'window': 5, 'min_count': 10, 
              'workers': max(1, multiprocessing.cpu_count() - 1), 'hs' : 0, 'sg' : 1, 'negative' : 10, 'iter' : 30, 'sample': 1e-3, 'seed' : 1}
    word2vec = Word2Vec(sentences, **params)
    word2vec.save(folder + '/' + "word2vec.model")