from reuters import *
from pandas import DataFrame
import unicodedata

# Get data from reuters folder.
if __name__ == "__main__":
    data_streamer = ReutersStreamReader('reuters').iterdocs()
    data = get_minibatch(data_streamer, 50000)

    data_all = DataFrame({'text': [], 'tags': []})
    rows_all = []

    for i in range(0, len(data.tags)):
        temp = data["text"][i]
        temp = str(temp)
        temp = str(unicodedata.normalize('NFKD', temp).encode('ascii', 'ignore'))
        # Replace characters.
        for character in ["\n", "|", ">", "<", "-", "+", "^", "[", "]", "#", "\t", "\r", "`"]:
            temp = temp.replace(character, " ")
        if not temp.isspace():
            rows_all.append({'text': temp, 'tags': data["tags"][i]})

    data_all = data_all.append(DataFrame(rows_all))
    data_all.to_pickle('all.pkl')
