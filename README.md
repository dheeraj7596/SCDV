# Text Classification with Sparse Composite Document Vectors


## Introduction
  - For text classification and information retrieval tasks, text data has to be represented as a fixed dimension vector. 
  - We propose simple feature construction technique named **Sparse Document Vectors (SDV).**
  - We demonstrate our method through experiments on multi-class classification on 20newsgroup dataset and multi-label text classification on Reuters-21578 dataset. 

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
Get Sparse Document Vectors (SDV) for documents in train and test set and accuracy of prediction on test set:
```sh
$ python SDV.py 200 60
# SDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
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
Get Sparse Document Vectors (SDV) for documents in train and test set:
```sh
$ python SDV.py 200 60
# SDV.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```
Get performance metrics on test set:
```sh
$ python metrics.py 200 60
# metrics.py takes word vector dimension and number of clusters as arguments. We took word vector dimension as 200 and number of clusters as 60.
```

## Requirements
Minimum requirements:
  -  Python 2.7+
  -  NumPy 1.8+
  -  Scikit-learn
  -  Pandas
  -  Gensim

For theory and explanation of SDV, please visit https://dheeraj7596.github.io/SDV/.

    Note: You neednot download 20Newsgroup or Reuters-21578 dataset. All datasets are present in their respective directories.

[//]: # (We used SGMl parser for parsing Reuters-21578 dataset from  https://gist.github.com/herrfz/7967781)


   [git-repo-url]: < https://gist.github.com/herrfz/7967781>
