# Baseline Model
This is a baseline model for [DSTC task 2](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling). This is a GRU-based seq2seq generation system. Since it is a baseline, the model does not use grounding information ("facts"), attention or beam search. It uses greedy decoding (unkown token disabled). The implementation is in Python, adapted from a Keras [tutorial](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html). 

## Requirement
The scripts are tested on [Python 3.6.4](https://www.python.org/downloads/) with the following libaries
* [Keras 2.1.6](https://keras.io/), which requires another backend lib. We used [Tensorflow 1.8.0](https://www.tensorflow.org/)
* [numpy 1.14.0](http://www.numpy.org/)

## Input files
Trial data extraction script and instructions are available [here](https://github.com/DSTC-MSR-NLP/DSTC7-End-to-End-Conversation-Modeling/tree/master/data_extraction). Based on the conversation text file, the following input files are generated with the command:
```
python create_input_files.py
```
|generated file|description|
|---|---|
|dict.txt|The vocab list. The line id is the `word id` (starting from 1) of the word in this line. The words are ordered by their frequencies appeared in the raw input file|
|source_num.txt|The list of source sentences where words are replaced by their `word id`|
|target_num.txt|The list of corresponding target sentences where words are replaced by their `word id`|

## Parameters
Key parameters can be specified in the `main()` function of [baseline.py](baseline.py). We follow [DSTC 2016](https://github.com/dialogtekgeek/DSTC6-End-to-End-Conversation-Modeling/blob/master/ChatbotBaseline/egs/twitter/run.sh) for the hyperparameter value settings.

|parameter|description|value|
|---------|-------|-----|
|`token_embed_dim` | length of word embedding vector |100|
|`rnn_units`| number of hidden units of each GRU cell|512|
|`encoder_depth`| number of GRU cells stacked in the encoder|2|
|`decoder_depth`| number of GRU cells stacked in the decoder|2|
|`dropout_rate`| dropout probability|0.5|
|`max_num_token`| if not None, only use top `max_num_token` most frequent tokens|20000|
|`max_seq_len`| tokens after the first `max_seq_len` tokens will be discarded |32|


## Run
Use the command:
```
python baseline.py [mode]
```
where  `mode` can be one of the following values

|mode|description|
|---------|-------|
|`train` | train a new model. The trained model is saved after each epoch |
|`continue` | load existing model and continue the training |
|`eval`| evaluate the model on held-out data. Negative log likelihood loss is printed|
|`interact`| explore the trained model interactively|
