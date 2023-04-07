## Complex Facial Expression Recognition Using Deep Knowledge Distillation of Basic Features

This repository contains code for training and evaluating the models described in the paper "Complex Facial Expression Recognition Using Deep Knowledge Distillation of Basic Features" by Angus Maiden and Bahareh Nakisa. The code is developed and maintained by Angus Maiden. The program can be run in three phases, corresponding with those described in the paper, as well as a visualisation phase:
- **Basic FER Phase**  
Train a model to recognise the basic emotions (Anger, Disgust, Fear, Happy, Sad, Surprise).
- **Continual Learning Phase**  
The trained model from the Basic FER Phase is used to learn new compound expression classes sequentially, by incrementally adding new classes until all of the expressions have been learned.
- **Few-Shot Learning Phase**  
The trained model from the Basic FER Phase is used to learn new compound expression classes, one at a time, using only a very small number of samples of the new class.
- **Visualisation**  
Produce the visualisations from the paper.

The dataset used for training is Compound Facial Expressions of Emotion (CFEE), which can be downloaded from https://cbcsl.ece.ohio-state.edu/compound.html. You will need to request access from the dataset provider. The unzipped dataset folder "CFEE_Database_230" should be placed in the "data/raw" folder. The program will save the processed data to the "data/processed" folder after face detection and alignment. The program will save the trained models to the "models" folder, results to the "results" folder, and images and visualisations to the "images" folder.

## Installation

Prerequisites:
Python 3.9

Download the repository and navigate to it:
```
git clone https://github.com/AngusMaiden/complex-FER.git
cd complex-FER
```
Create a new python environment:
```
pip install venv
python -m venv complex-FER
```
Activate the environment  
For Linux:
```
source env/bin/activate
```
(or) For Windows:  
```
.\env\Scripts\activate
```
Install the program:
```
pip install .
```

# Usage

The code in this repository is designed to be run as a self-contained command line interface (CLI).

To run the experiments in different phases and with different hyperparameters, run this command from within the `complex-FER` directory from the installation steps above:
```
complex-FER <options>
```

You can see a list of options and their usage by running:
```
complex-FER --help
```

```
Options:
  --phase TEXT                   Phase of the experiment to run. Options: all,
                                 basic, cont, fewshot, vis. Default: all
  --seed INTEGER                 Seed to use for random number generation.
                                 Default: 42. Enter "None" for no random seed.
  --dataset TEXT                 Dataset to use for training. Options: CFEE
  --epochs INTEGER               Number of epochs to train for. Default: 1000
  --basic_batch_size INTEGER     Batch size to use for training in the Basic
                                 FER Phase. Default: 32
  --cont_batch_size INTEGER      Batch size to use for training in the
                                 Continual Learning Phase. Default: 16
  --fewshot_batch_size INTEGER   Batch size to use for training in the Few-
                                 Shot Learning Phase. Default: 32
  --basic_lr FLOAT               Learning rate to use for training in the
                                 Basic FER Phase. Default: 1e-4
  --cont_lr FLOAT                Learning rate to use for training in the
                                 Continual Learning Phase. Default: 1e-5
  --basic_finetuning_lr FLOAT    Learning rate to use for finetuning in the
                                 Basic FER Phase. Default: 1e-6
  --patience INTEGER             Patience to use for early stopping. Default:
                                 100
  --basic_frozen_layers INTEGER  Number of layers to freeze for transfer
                                 learning in Basic FER Phase. Default: 86
  --cont_frozen_layers INTEGER   Number of layers to freeze for transfer
                                 learning in Continual Learning Phase.
                                 Default: 154
  --cont_fold INTEGER            Complex emotion list to use in Continual
                                 Learning Phase. Default: 0
  --cont_val_fold INTEGER        Validation fold from Basic FER Phase to use
                                 in Continual Learning Phase. Default: 0
  --cont_mem_mode TEXT           Representative Memory mode to use for
                                 Continual Learning Phase. Options: limit,
                                 grow. Default: grow
  --img_height INTEGER           Height of images to use for training.
                                 Default: 224
  --img_width INTEGER            Width of images to use for training. Default:
                                 224
  --num_basic_classes INTEGER    Number of basic classes to use for training.
                                 Default: 6
  --exclude_emotions TEXT        Comma-separated list of emotions to exclude
                                 from training. Default: Neutral
  --n_shots INTEGER              Number of shots to use for Few-shot Learning
                                 Phase. Default: 5. Enter "all" to use all
                                 training data.
  --subj INTEGER                 Subject to use for visualisation. Default: 1.
  --help                         Show this message and exit.
  ```
