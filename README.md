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
- train Naive Bayes classifier with BoW + TF-IDF word embedding
```console
foo@bar:~$ python src/train.py -d data/dataset.txt -t 0.1 -c nb -w bow -o artifact
```
- train SVM classifier with GLoVe-50d word embedding
```console
foo@bar:~$ python src/train.py -d data/dataset.txt -t 0.1 -c svm -w glove -o artifact
```
- train deep CNN classifier with GLoVe-50d word embedding
```console
foo@bar:~$ python src/dnn_train.py -d data/dataset.txt -t 0.1 -c cnn -w glove-50 -o artifact
```
- train deep CNN-LSTM classifier with GLoVe-50d word embedding
```console
foo@bar:~$ python src/dnn_train.py -d data/dataset.txt -t 0.1 -c lstm -w glove-50 -o artifact
```
- train deep CNN-GRU classifier with GLoVe-50d word embedding
```console
foo@bar:~$ python src/dnn_train.py -d data/dataset.txt -t 0.1 -c gru -w glove-50 -o artifact
```


### Predict:
```console
foo@bar:~$ python src/predict.py -m models/nb.pkl -d models/word_dictionary.json -e data/emoji_map_1791.csv -s "I am happy"
```

### Accuracy
|  Word Embedding |  BoW + TF-IDF |  GLoVe-50d  | GLoVe-300d  |     BERT    |
|-----------------|---------------|-------------|-------------|-------------|
| Naive Bayes     |  20.134%      |     N/A     |   N/A       |     N/A     |
| SVM             |  8.255%       | 14.497%     | 14.430%     |
| CNN             |    N/A        | 14.765%     | 15.638%     |
| LSTM            |    N/A        | 14.295%     |             |
| LSTM + Attention|    N/A        |             |             |


### To-do List
- [ ] implement BERT as word-embedding
- [ ] implement attention mechanism
