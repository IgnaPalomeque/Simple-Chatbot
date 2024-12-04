# Simple Chatbot
## Simple Chatbot example using TensorFlow and the [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset
<br>

### Model summary:

| Layer (type)                  | Output Shape                                 | Param #         | Connected to                                     |
--------------------------------|---------------------------                   |-----------------|--------------------------------------------------|
| `input_layer (InputLayer)`    | **(None, 572)**                              | **0**           | **-**                                            |
|                               |                                              |                 |                                                  |
| `input_layer_1 (InputLayer)`  | **(None, 572)**                              | **0**           | **-**                                            |
|                               |                                              |                 |                                                  |
| `embedding (Embedding)`       | **(None, 572, 256)**                         | **27,130,368**  | **input_layer[0][0]**                            |
|                               |                                              |                 |                                                  |
| `embedding_1 (Embedding)`     | **(None, 572, 256)**                         | **27,130,368**  | **input_layer_1[0][0]**                          |
|                               |                                              |                 |                                                  |
| `lstm (LSTM)`                 | **[(None, 256), (None, 256), (None, 256)]**  | **525,312**     | **embedding[0][0]**                              |
|                               |                                              |                 |                                                  |
| `lstm_1 (LSTM)`               | **(None, 572, 256)**                         | **525,312**     | **embedding_1[0][0], lstm[0][1], lstm[0][2]**    |
|                               |                                              |                 |                                                  |
| `dense (Dense)`               | **(None, 572, 105978)**                      | **27,236,346**  | **lstm_1[0][0]**                                 |


<br><br>
This is just one of many projects that will document my learning journey with TensorFlow.
