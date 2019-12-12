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
- train [NB / SVM] classifier with [bow / GLoVe] word embedding
```console
foo@bar:~$ python src/train.py -d $DATAST_PATH -t $TEST_RATIO -c $CLASSFIER_TYPE -w $WORD_EMBEDDING_TYPE -o $ARTIFACT
```
- train [CNN / LSTM / GRU] classifier with [GLoVe-50d / GLoVe-300d / BERT] word embedding
```console
foo@bar:~$ python src/dnn_train.py -d $DATAST_PATH -t $TEST_RATIO -c $CLASSFIER_TYPE -w $WORD_EMBEDDING_TYPE -o $ARTIFACT
```

### Predict:
```console
foo@bar:~$ python src/predict.py -m models/nb.pkl -d models/word_dictionary.json -e data/emoji_map_1791.csv -s "I am happy"
```

### Accuracy
|  Word Embedding |  BoW + TF-IDF |  GLoVe-50d  | GLoVe-300d  |     BERT    |
|-----------------|---------------|-------------|-------------|-------------|
| Naive Bayes     |  19.530%      |     N/A     |   N/A       |     N/A     |
| SVM             |  9.195%       | 16.376%     | 14.966%     |             |
| CNN             |    N/A        | 15.168%     | 15.906%     |             |
| LSTM            |    N/A        | 15.705%     | 15.570%     |             |
| LSTM + Attention|    N/A        |             |             |             |


### To-do List
- [ ] implement BERT as word-embedding
- [ ] implement attention mechanism
