# Text Classification with Sparse Composite Document Vectors (SCDV)


## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose simple feature construction technique named [**SCDV: Sparse Composite Document Vectors using soft clustering over distributional representations.**](https://www.aclweb.org/anthology/D17-1069.pdf) presented at EMNLP 2017.
  - We demonstrate our method through experiments on multi-class classification on 20newsgroup dataset and multi-label text classification on Reuters-21578 dataset. 

## Citation
If you find SCDV useful in your research, please consider citing:
```
@inproceedings{mekala2017scdv,
  title={SCDV: Sparse Composite Document Vectors using soft clustering over distributional representations},
  author={Mekala, Dheeraj and Gupta, Vivek and Paranjape, Bhargavi and Karnick, Harish},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={659--669},
  year={2017}
}
```

## Testing
There are 2 folders named 20news and Reuters which contains code related to multi-class classification on 20Newsgroup dataset and multi-label classification on Reuters dataset.

#### 20Newsgroup
Change directory to 20news for experimenting on 20Newsgroup dataset and create train and test tsv files as follows:
```sh
$ cd 20news
$ python create_tsv.py
```
Get word vectors for all words in vocabulary:
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set and accuracy of prediction on test set:
```sh
$ python SCDV.py 200 60
# SCDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
Get Topic coherence for documents in train set:
```sh
$ python TopicCoherence.py 200 60
# TopicCoherence.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
#### Reuters
Change directory to Reuters for experimenting on Reuters-21578 dataset. As reuters data is in SGML format, parsing data and creating pickle file of parsed data can be done as follows:
```sh
$ python create_data.py
# We don't save train and test files locally. We split data into train and test whenever needed.
```
Get word vectors for all words in vocabulary: 
```sh
$ python Word2Vec.py 200
# Word2Vec.py takes word vector dimension as an argument. We took it as 200.
```
Get Sparse Document Vectors (SCDV) for documents in train and test set:
```sh
$ python SCDV.py 200 60
# SCDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
Get performance metrics on test set:
```sh
$ python metrics.py 200 60
# metrics.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```

#### Information Retrieval
Change directory to IR for experimenting on information Retrieval task. IR Datasets mentioned in the paper can be downloaded from [TREC website](http://trec.nist.gov/data/docs_eng.html). 

You will need to run the documents and queries through a full fledged IR pipeline system like Apache Lucene or [Project Lemur](https://www.lemurproject.org/) in order to 
  - Tokenize the data, remove stop words and pass tokens through a Porter Stemmer.
  - Build inverted and forward index.
  - Build a basic language model retrieval system with Dirichlet smoothing.

Data Format
  - The IR Data folder must have a file called "queries.txt" and a folder called *raw* that has all the documents.
  - Each file in *raw* should be a single document containing space separated processed tokens. File must be named as doc_ID.txt.
  - Each line in queries.txt should be a single query containing space separated processed words.

To interpolate language model retrieval system with the query-document score obtained from SCDV:

Get word vectors for all terms in vocabulary:
```sh
$ python Word2Vec.py 300 sjm
# Word2Vec.py takes word vector dimension and folder containing IR dataset as arguments. We took 300 and sjm (San Jose Mercury).
```
Create Sparse Document Vectors (SCDV) for all documents and queries and compute similarity scores for all query-document pairs.
```sh
$ python SCDV.py 300 100 sjm
# SCDV.py takes word vector dimension, number of clusters as arguments and folder containing IR dataset as arguments. We took 300 100 and sjm.
# Change the code to store these scores in a format that can be used by the IR system.
```
Use these scores to interpolate with the language model scores with interpolation parameter 0.5.


## Requirements
Minimum requirements:
  -  Python 2.7+
  -  NumPy 1.8+
  -  Scikit-learn
  -  Pandas
  -  Gensim

For theory and explanation of SCDV, please visit our [EMNLP 2017 paper](https://www.aclweb.org/anthology/D17-1069.pdf).

Note: You neednot download 20Newsgroup or Reuters-21578 dataset. All datasets are present in their respective directories. We used SGMl parser for parsing Reuters-21578 dataset from [here](https://gist.github.com/herrfz/7967781)
