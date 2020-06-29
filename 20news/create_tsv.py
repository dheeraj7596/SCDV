from sklearn.datasets import fetch_20newsgroups
from pandas import DataFrame
import unicodedata

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

rows_train = []
rows_test = []
rows_all = []

# Path for saving all tsv files.
path_train = 'data/train_v2.tsv'
path_test = 'data/test_v2.tsv'
path_all = 'data/all_v2.tsv'

data_train = DataFrame({'news': [], 'class': []})
data_test = DataFrame({'news': [], 'class': []})
data_all = DataFrame({'news': []})

for i in range(0, len(newsgroups_train.data)):
    target = newsgroups_train.target[i]
    # Convert into unicode
    newsgroups_train.data[i] = str(newsgroups_train.data[i])
    # Remove characters which can't be converted into ascii.
    newsgroups_train.data[i] = str(unicodedata.normalize('NFKD', newsgroups_train.data[i]).encode('ascii', 'ignore'))
    for character in ["\n", "|", ">", "<", "-", "+", "^", "[", "]", "#", "\t", "\r", "`"]:
        newsgroups_train.data[i] = newsgroups_train.data[i].replace(character, " ")

    rows_train.append({'news': newsgroups_train.data[i], 'class': target})
    rows_all.append({'news': newsgroups_train.data[i]})

for i in range(0, len(newsgroups_test.data)):
    target = newsgroups_test.target[i]
    # Convert into unicode
    newsgroups_test.data[i] = str(newsgroups_test.data[i])
    # Remove characters which can't be converted into ascii.
    newsgroups_test.data[i] = str(unicodedata.normalize('NFKD', newsgroups_test.data[i]).encode('ascii', 'ignore'))
    for character in ["\n", "|", ">", "<", "-", "+", "^", "[", "]", "#", "\t", "\r", "`"]:
        newsgroups_test.data[i] = newsgroups_test.data[i].replace(character, " ")

    rows_test.append({'news': newsgroups_test.data[i], 'class': target})
    rows_all.append({'news': newsgroups_test.data[i]})

# Dump into tsv's.
data_train = data_train.append(DataFrame(rows_train))
data_train.to_csv(path_or_buf=path_train, sep='\t', encoding="utf-8")

data_test = data_test.append(DataFrame(rows_test))
data_test.to_csv(path_or_buf=path_test, sep='\t', encoding="utf-8")

data_all = data_all.append(DataFrame(rows_all))
data_all.to_csv(path_or_buf=path_all, sep='\t', encoding="utf-8")
