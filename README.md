# DeepMoji
CS229 course project deep learning model for analyzing sentiment, emotion, sarcasm etc

### Download GLoVe Pre-trained model
```console
foo@bar:~$ bash models/download_glove.sh
```

### Download NLTK stop words
```console
foo@bar:~$ python -m nltk.downloader all
```

### Train:
##### train Naive Bayes classifier with BoW + TF-IDF word embedding
```console
foo@bar:~$ python src/train.py -d data/dataset.txt -t 0.1 -c nb -w bow -s models
```
##### train SVM classifier with BoW + GLoVe-50d word embedding
```console
foo@bar:~$ python src/train.py -d data/dataset.txt -t 0.1 -c svm -w glove -s models
```

### Predict:
```console
foo@bar:~$ python src/predict.py -m models/nb.pkl -d models/word_dictionary.json -e data/emoji_map_1791.csv -s "I am happy"
```

### Accuracy
|  Word Embedding |  BoW + TF-IDF |  GLoVe-50d  | GLoVe-300d  |     BERT    |
|-----------------|---------------|-------------|-------------|-------------|
| Naive Bayes     |  16.713%      |     N/A     |   N/A       |     N/A     |
| SVM             |  14.680%      | 15.048%     | 14.873%     |
| CNN             |               |             |             |
| GRU             |               |             |             |
| GRU + Attention |               |             |             |


### To-do List
- [ ] textCNN with pre-trained GLoVe
- [ ] textCNN with pre-trained BERT
