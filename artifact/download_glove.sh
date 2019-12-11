#!/bin/bash

mkdir artifact/pretrain_models
wget http://nlp.stanford.edu/data/glove.6B.zip -P artifact/pretrain_models
unzip artifact/pretrain_models/glove.6B.zip -d artifact/pretrain_models
rm artifact/pretrain_models/glove.6B.zip