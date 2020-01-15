In this folder are all training sets.

The columns in the table represent: original text, cleaned text, cui corresponding to the text

- ./PICO_Heartdisease
heart disease dataset. 
This is the dataset marked by the experts in the initial stage with reference to the systematic review.


- ./PICO_Orthopedic
Orthopedics dataset.


- ./PICO_EBMNLP

This is a training set based on the ebm-nlp dataset. We combine consecutive identically labeled words in sentences in the original data set into phrases as training samples.

Aceso's input needs words and corresponding concepts. Input files in ./PICO_EBMNLP have generated concepts using MetaMap.