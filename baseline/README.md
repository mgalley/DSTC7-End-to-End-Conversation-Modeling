# Baseline Model
This is a baseline model for [DSTC task 2](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling). This is a GRU-based seq2seq generation system. Since it is a baseline, the model does not use grounding information ("facts"), attention or beam search. The implementation is in Python, adapted from a Keras [tutorial](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html). 

## Requirement
The scripts are tested on [Python 3.6](https://www.python.org/downloads/) with the following libaries
* [Keras](https://keras.io/), which requires another backend lib. We used [Tensorflow](https://www.tensorflow.org/)
* [numpy](http://www.numpy.org/)

## Input files
Trial data files will be available soon [(here)](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction). The dataset will include the following input files.

|generated file|description|
|---|---|
|dict.txt|The vocab list. The line number is the `word id` (starting from 1) of the word in this line. The words are ordered by their frequencies appeared in the raw input file|
|source_num.txt|The list of source sentences where words are replaced by their `word id`|
|target_num.txt|The list of corresponding target sentences where words are replaced by their `word id`|

## Parameters
Key parameters can be specified in the main() function of [baseline.py](baseline.py)

|parameter|description|
|---------|-------|
|`token_embed_dim` | length of word embedding vector |
|`rnn_units`| number of hidden units of each GRU cell|
|`encoder_depth`| number of GRU cells stacked in the encoder|
|`decoder_depth`| number of GRU cells stacked in the decoder|
|`dropout_rate`| dropout probability|
|`learning_rate`| learning rate|

## Run
Use the command:
```
python baseline.py [mode]
```
where mode can be one of the following values

|mode|description|
|---------|-------|
|`train` | train the model on randomly selected training data. The trained model is saved after each epoch |
|`test`| test the model on held-out data. Negative likelihood is printed|
|`interact`| explore the trained model interactively|
