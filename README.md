# DeepMoji
CS229 course project deep learning model for analyzing sentiment, emotion, sarcasm etc

### Training:
```console
foo@bar:~$ python src/main.py -d data/dataset.txt -c nb -n 20000 -s models
```

### Inference:
```console
foo@bar:~$ python src/demo.py -m models/nb.pkl -d models/word_dictionary.json -e data/emoji_map_1791.csv -s "I am happy"
```

### Accuracy
| Classifier  |  Accuracy |
|-------------|-----------|
| Naive Bayes |  13.815%  |
| SVM         |  12.009%  |
| DNN         |   |
