# ACESO

![license](https://img.shields.io/github/license/mashape/apistatus.svg)    ![build](https://img.shields.io/teamcity/http/teamcity.jetbrains.com/s/bt345.svg)       ![metamap](https://img.shields.io/badge/MetaMap-2016v2-brightgreen.svg)

## More About ACESO

This is the codes of the article ACESO: PICO-guided Evidence Summarization on Medical Literature, it includes all experimental codes and datasets.

### Project Structs

- data clean
- data transform
- data train


## Requirements

* python 3.6
* pytorch 0.3.1
* visdom 0.1.8.5 
* torchnet 0.0.4
* sklearn 0.20.1
* nltk 3.3.0
* gensim  3.4.0
* tqdm 4.28.1
* fire 0.1.3
* pandas 0.23.4


## Datasets
in the file of datasets/PICO/:

* P.csv 280

* I.csv 290

* O.csv 320

* N.csv 330

## run and test
1. start the visdom server :  python -m visdom.server
2. train: update the config.py and write the data location,then python main.py train
3. test: update the config.py and then, python main.py test

### Visualize

please read the document about [Visidom](https://github.com/facebookresearch/visdom)

 ### Results

![results](https://raw.githubusercontent.com/wen-fei/ACESO/master/images/results.png?token=AIcBSQVqYCV6FRhFWTRCIAmTTFuHIPlkks5cBMtEwA%3D%3D)

