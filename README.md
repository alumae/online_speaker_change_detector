# Online (streaming) speaker change detection model implemented in Pytorch

This repository contains an implementation of an online streaming speaker change detection model.
It is implemented in Pytorch.

The model consists of several 1-D convolutional layers acting as speech encoder,
a multi-layer LSTM that models speaker change, and a final softmax layer. The model uses
a step size of 100 ms (i.e., it outputs 10 decisions per second).

The model is trained using a special version of cross-entropy
training which tolerates small errors in the hypthesized speaker change timestamps.
Due to this, the softmax outputs of the trained model are very peaky and do not require
any local maxima tracking for extracting the final speaker turn points. This
makes the model suitable for online appications.

The test directory contains a model trained on Estonian broadcast data.

## Demo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alumae/online_speaker_change_detector/blob/main/tutorials/streaming_demo.ipynb)
