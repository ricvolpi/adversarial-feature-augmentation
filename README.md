# Code for the paper Adversarial Feature Augmentation for Unsupervised Domain Adaptation

Step 0: training a classifier on source data.

Step 1: training a feature generator to perform feature augmentation in the source feature space.

(WIP) Step 2: training an encoder by adapting it to the source features, relying on the pre-trained feature generator.

## Overview

### Files

Model.py: contains the models described in the paper, implemented in Tensorflow (slim)

TrainOps.py: contains the operations to perform Step 0 and Step 1, described in the paper. (Step 2 WIP)

### Prerequisites

Python 2.7, Tensorflow 1.3

## How it works

To obtain the MNIST dataset, run

```
python download_and_process_mnist.py
```

To train a ConvNet feature extractor, run

```
python main.py --mode=train_feature_extractor --gpu=0
```

To train a generator of features that resemble the one extracted through the pre-trained feature extractor, run

```
python main.py --mode=train_feature_generator --gpu=0
```

Of course, you choose the GPU index. The resulting model (feature_generator) can be used to generate new features from the desired classes, by feeding it with noise vectors concatenated with one-hot label codes.

## What to expect

In all our experiments, carried out on six different datasets, we could observe the same pattern while training the feature generator: the discriminator will seem to win the minimax game for the first few thousand iterations, then the loss associated to the generator will start to decrease. In every experiment we ran, we ended up with a very stable stability point, with the discriminator giving as output numbers very close to 0.5 independently by the input given (real/generated features).

