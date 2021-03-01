# Online (streaming) speaker change detection model implemented in Pytorch

This repository contains an implementation of an online streaming speaker change detection model.
It is implemented in Pytorch.

The model consists of several 1-D convolutional layers acting as speech encoder,
a multi-layer LSTM that model speaker change, and a final softmax layer.

The test directory contains a model trained on Estonian broadcast data.