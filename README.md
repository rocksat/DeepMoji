# DeepMoji
CS229 course project deep learning model for analyzing sentiment, emotion, sarcasm etc

### Train:
```console
foo@bar:~$ python src/train.py -d data/dataset.txt -c nb -n 20000 -s models
```

### Predict:
```console
foo@bar:~$ python src/predict.py -m models/nb.pkl -d models/word_dictionary.json -e data/emoji_map_1791.csv -s "I am happy"
```

### Accuracy
|             |  Bag of Word |  GLoVe-50d  | GLoVe-300d  |
|-------------|--------------|-------------|-------------|
| Naive Bayes |  16.713%     |     N/A     |   N/A       |
| SVM         |  14.680%     | 15.048%     | 14.873%     |
| LSTM        |              |             |             |
