#!/usr/bin/env python
# coding: utf-8

## Library Imports and System Functions
import os
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import gc
import math
import click
import pickle
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, Sequential
from tensorflow.keras.utils import set_random_seed
from tensorflow.config.experimental import list_physical_devices, list_logical_devices, set_memory_growth
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

gpus = list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
        logical_gpus = list_logical_devices('GPU')
        print(f'Found GPU on system: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e:
       #virtual devices must be set before GPUs have been initialized
        print(e)
else:
    print('No GPU found on system')

from . import constants
from .data import utils, process
from .models import train, evaluate
from .visualisation import visualise

@click.command()
@click.option('--phase', default='all', help='Phase of the experiment to run.\nOptions: all, basic, cont, fewshot, vis. Default: all')
@click.option('--seed', default=42, help='Seed to use for random number generation.\nDefault: 42. Enter "None" for no random seed.')
@click.option('--dataset', default='CFEE', help='Dataset to use for training.\nOptions: CFEE')
@click.option('--epochs', default=1000, help='Number of epochs to train for.\nDefault: 1000')
@click.option('--basic_batch_size', default=32, help='Batch size to use for training in the Basic FER Phase.\nDefault: 32')
@click.option('--cont_batch_size', default=16, help='Batch size to use for training in the Continual Learning Phase.\nDefault: 16')
@click.option('--fewshot_batch_size', default=32, help='Batch size to use for training in the Few-Shot Learning Phase.\nDefault: 32')
@click.option('--basic_lr', default=1e-4, help='Learning rate to use for training in the Basic FER Phase.\nDefault: 1e-4')
@click.option('--cont_lr', default=1e-5, help='Learning rate to use for training in the Continual Learning Phase.\nDefault: 1e-5')
@click.option('--basic_finetuning_lr', default=1e-6, help='Learning rate to use for finetuning in the Basic FER Phase.\nDefault: 1e-6')
@click.option('--patience', default=100, help='Patience to use for early stopping.\nDefault: 100')
@click.option('--basic_frozen_layers', default=86, help='Number of layers to freeze for transfer learning in Basic FER Phase.\nDefault: 86')
@click.option('--cont_frozen_layers', default=154, help='Number of layers to freeze for transfer learning in Continual Learning Phase.\nDefault: 154')
@click.option('--cont_fold', default=0, help='Complex emotion list to use in Continual Learning Phase.\nDefault: 0')
@click.option('--cont_val_fold', default=0, help='Validation fold from Basic FER Phase to use in Continual Learning Phase.\nDefault: 0')
@click.option('--cont_mem_mode', default='grow', help='Representative Memory mode to use for Continual Learning Phase.\nOptions: limit, grow. Default: grow')
@click.option('--img_height', default=224, help='Height of images to use for training.\nDefault: 224')
@click.option('--img_width', default=224, help='Width of images to use for training.\nDefault: 224')
@click.option('--num_basic_classes', default=6, help='Number of basic classes to use for training.\nDefault: 6')
@click.option('--exclude_emotions', default='Neutral', help='Comma-separated list of emotions to exclude from training.\nDefault: Neutral')
@click.option('--n_shots', default=5, help='Number of shots to use for Few-shot Learning Phase.\nDefault: 5. Enter "all" to use all training data.')
@click.option('--subj', default=1, help='Subject to use for visualisation.\nDefault: 1.')
def main(
    phase,
    seed,
    dataset,
    epochs,
    basic_batch_size,
    cont_batch_size,
    fewshot_batch_size,
    basic_lr,
    cont_lr,
    basic_finetuning_lr,
    patience,
    basic_frozen_layers,
    cont_frozen_layers,
    cont_fold,
    cont_val_fold,
    cont_mem_mode,
    img_height,
    img_width,
    num_basic_classes,
    exclude_emotions,
    n_shots,
    subj
    ):
    '''
    A program to train and evaluate the models described in the paper "Complex Facial Expression Recognition Using Deep Knowledge Distillation of Basic Features" by A. Maiden and B. Nakisa. The program can be run in three phases, corresponding with those described in the paper, as well as a visualisation phase:
    
    1. Basic FER Phase: Train a model to recognise the basic emotions (Anger, Disgust, Fear, Happy, Sad, Surprise).
    2. Continual Learning Phase: The trained model from the Basic FER Phase is used to learn new compound expression classes sequentially, by incrementally adding new classes until all of expressions have been learned.
    3. Few-Shot Learning Phase: The trained model from the Basic FER Phase is used to learn new compound expression classes, one at a time, using only a very small number of samples of the new class.
    4. Visualisation: Produce the visualisations from the paper.

    The dataset used for training is the CFEE Database 230, which can be downloaded from https://cbcsl.ece.ohio-state.edu/compound.html. You will need to request access from the dataset provider. The unzipped dataset folder "CFEE_Database_230" should be placed in the "data/raw" folder. The program will save the processed data to the "data/processed" folder after face detection and alignment. The program will save the trained models to the "models" folder, results to the "results" folder, and images and visualisations to the "images" folder.
    '''
    if str(seed).lower() != 'none':
        set_random_seed(seed)

    if dataset == 'CFEE':
        dataset_dir = 'CFEE_Database_230'
    data_dir = constants.processed_data_dir / 'CFEE_Database_230'

    if (constants.interim_data_dir / 'annotation_dict.pkl').exists():
        with open(constants.interim_data_dir / 'annotation_dict.pkl', 'rb') as f:
            annotation_dict = pickle.load(f)
    else:
        annotation_dict = pd.read_excel(data_dir / 'AU_annotation_all_subjects.xls', sheet_name = None)
        for k, v in annotation_dict.items():
            annotation_dict[k] = v.fillna(0)
        with open('annotation_dict.pkl', 'wb') as f:
            pickle.dump(annotation_dict, f)

    emotion_list = annotation_dict['summary'].iloc[:,1].to_list()
    emotion_ids = annotation_dict['summary'].iloc[:,0].to_list()

    if exclude_emotions:
        excluded_emotions = exclude_emotions.split(',')

        emotion_ids = [emotion_ids[emotion_list.index(i)] for i in emotion_list if i not in excluded_emotions]
        emotion_list = [i for i in emotion_list if i not in excluded_emotions]

    emotion_ids = [f'{emotion_id:02d}' for emotion_id in emotion_ids]

    basic_emotion_list = emotion_list[:num_basic_classes]
    basic_emotion_ids = emotion_ids[:num_basic_classes]

    try:
        next(data_dir.glob('*/Images*/*_detectface.[jJ][pP][gG]'))
    except Exception:
        print('No face detection images found. Running face detection...')
        process.detectface(emotion_ids, constants.raw_data_dir / dataset_dir)
        
    file_list = [str(f) for f in data_dir.glob('*/Images*/*_detectface.[jJ][pP][gG]') \
                 if any([idx == os.path.split(f)[1][:2] for idx in emotion_ids])]

    label_list = []
    for file in file_list:
        label_list.append(os.path.split(file)[1][:2])

    subj_list = sorted(list(set([os.path.split(f)[1][3:6] for f in file_list])))

    fold_length = int(len(subj_list)/10)

    file_list = np.array(file_list)
    label_list = np.array(label_list)

    es_loss_callback = EarlyStopping(monitor = 'val_loss', patience = patience, min_delta = 0.01, restore_best_weights = True)
    es_acc_callback = EarlyStopping(monitor = 'val_accuracy', patience = patience, restore_best_weights = True)

    data_augmentation = Sequential([layers.RandomFlip("horizontal"),
                                    layers.RandomRotation(0.01),
                                    layers.RandomZoom(0.05, 0.05),
                                    layers.RandomTranslation(0.05, 0.05)], name = 'data_augmentation')

    if phase == 'all':
        cont_val_fold = basic(
            num_basic_classes,
            basic_emotion_list,
            basic_emotion_ids,
            epochs,
            basic_batch_size,
            basic_lr,
            basic_finetuning_lr,
            file_list,
            subj_list,
            fold_length,
            img_height,
            img_width,
            basic_frozen_layers,
            es_loss_callback,
            patience,
            data_augmentation
        )
        for cont_fold in range(10):
            cont(
                num_basic_classes,
                emotion_list,
                emotion_ids,
                basic_emotion_list,
                basic_emotion_ids,
                epochs,
                cont_batch_size,
                cont_lr,
                file_list,
                subj_list,
                fold_length,
                img_height,
                img_width,
                cont_frozen_layers,
                es_acc_callback,
                patience,
                data_augmentation,
                excluded_emotions,
                cont_fold,
                cont_val_fold,
                cont_mem_mode,
        )
        evaluate.get_overall_cont_results(num_basic_classes, cont_val_fold, excluded_emotions)
        for n_shots in (5,3,1):
            fewshot(
                num_basic_classes,
                emotion_list,
                emotion_ids,
                basic_emotion_list,
                basic_emotion_ids,                
                epochs,
                fewshot_batch_size,
                cont_lr,
                file_list,
                subj_list,
                fold_length,
                img_height,
                img_width,
                cont_frozen_layers,
                es_acc_callback,
                patience,
                data_augmentation,
                cont_val_fold,
                n_shots
        )
        vis(
        emotion_list,
        emotion_ids,
        basic_emotion_list,
        basic_emotion_ids,
        subj_list,
        file_list,
        num_basic_classes,
        img_height,
        img_width,
        cont_val_fold,
        subj
        )
    elif phase == 'basic':
        basic(
            num_basic_classes,
            basic_emotion_list,
            basic_emotion_ids,
            epochs,
            basic_batch_size,
            basic_lr,
            basic_finetuning_lr,
            file_list,
            subj_list,
            fold_length,
            img_height,
            img_width,
            basic_frozen_layers,
            es_loss_callback,
            patience,
            data_augmentation
        )
    elif phase == 'cont':
        cont(
            num_basic_classes,
            emotion_list,
            emotion_ids,
            basic_emotion_list,
            basic_emotion_ids,
            epochs,
            cont_batch_size,
            cont_lr,
            file_list,
            subj_list,
            fold_length,
            img_height,
            img_width,
            cont_frozen_layers,
            es_acc_callback,
            patience,
            data_augmentation,
            excluded_emotions,
            cont_fold,
            cont_val_fold,
            cont_mem_mode,
        )
    elif phase == 'fewshot':
        fewshot(
            num_basic_classes,
            emotion_list,
            emotion_ids,
            basic_emotion_list,
            basic_emotion_ids,
            epochs,
            fewshot_batch_size,
            cont_lr,
            file_list,
            subj_list,
            fold_length,
            img_height,
            img_width,
            cont_frozen_layers,
            es_acc_callback,
            patience,
            data_augmentation,
            cont_val_fold,
            n_shots
    )
    elif phase == 'vis':
        vis(
        emotion_list,
        emotion_ids,
        basic_emotion_list,
        basic_emotion_ids,
        subj_list,
        file_list,
        num_basic_classes,
        img_height,
        img_width,
        cont_val_fold,
        subj
        )
    else:
        print('Invalid phase. Options: all, basic, cont, fewshot')

def basic(
        num_basic_classes,
        basic_emotion_list,
        basic_emotion_ids,
        epochs,
        basic_batch_size,
        basic_lr,
        basic_finetuning_lr,
        file_list,
        subj_list,
        fold_length,
        img_height,
        img_width,
        basic_frozen_layers,
        es_loss_callback,
        patience,
        data_augmentation
        ):
    print('----Basic FER Phase----')
    print('-----------------------\n')

    if (constants.basic_results_dir / 'basic_results.pkl').exists():
        with open(constants.basic_results_dir / 'basic_results.pkl', 'rb') as f:
            basic_results = pickle.load(f)
            save = False
    else:
        save = True
        basic_results = {
        'history' : [],
        'finetuning_history' : [],
        'cm' : [],
        'accuracy' : [],
        'report' : [],
        'train_time' : [],
        'n_epochs' : []
        }

        for val_fold in range(10):

            print(f'Training Basic FER model (Validation fold: {val_fold})')

            # Generate files lists using the current validation fold.
            train_file_list, val_file_list, basic_train_file_list, basic_val_file_list = process.generate_file_lists(
                basic_emotion_ids,
                file_list,
                subj_list,
                fold_length,
                val_fold
                )

            # Generate datasets using the current validation fold.
            basic_train_ds = tf.data.Dataset.from_tensor_slices(basic_train_file_list).map(lambda x: process.process_path(x, basic_emotion_ids, num_basic_classes, img_height, img_width))\
                                                                                      .shuffle(len(basic_train_file_list))\
                                                                                      .batch(basic_batch_size)\
                                                                                      .map(lambda x, y: (data_augmentation(x, training = True), y))\
                                                                                      .prefetch(1)

            basic_val_ds = tf.data.Dataset.from_tensor_slices(basic_val_file_list).map(lambda x: process.process_path(x, basic_emotion_ids, num_basic_classes, img_height, img_width))\
                                                                                  .batch(basic_batch_size)\
                                                                                  .prefetch(1)

            # Build Basic FER model.
            if (constants.basic_models_dir / f'basic_model_{val_fold}').exists():
                basic_model = models.load_model(constants.basic_models_dir / f'basic_model_{val_fold}')
            else:
                basic_model = train.build_basic_model(num_basic_classes, img_height, img_width, basic_lr)

            # Train Basic FER model.
            t = time.process_time()
            basic_model, basic_history, basic_finetuning_history = train.train_basic_model(
                basic_model,
                basic_train_ds,
                basic_val_ds,
                epochs,
                es_loss_callback,
                basic_frozen_layers,
                basic_finetuning_lr,
                save_model_dir = constants.basic_models_dir / f'basic_model_{val_fold}')
            basic_train_time = time.process_time() - t
            print(f'Training time: {basic_train_time}')

            basic_n_epochs = len(basic_history['loss'])-patience+len(basic_finetuning_history['loss'])-patience

            # Evaluate FER model.
            evaluate.plot_training_history(
                basic_history,
                basic_finetuning_history,
                filename = constants.basic_images_dir / f'basic_training_history_{val_fold}.jpg',
                patience = patience
                )
            basic_cm, basic_accuracy, basic_report = evaluate.evaluate_model(
                basic_model,
                basic_val_ds,
                basic_emotion_list,
                constants.basic_images_dir / f'basic_cm_{val_fold}.jpg'
                )

            # Add results to evaluation and training history lists.
            basic_results['history'].append(basic_history)
            basic_results['finetuning_history'].append(basic_finetuning_history)
            basic_results['cm'].append(basic_cm)
            basic_results['accuracy'].append(basic_accuracy)
            basic_results['report'].append(basic_report)
            basic_results['train_time'].append(basic_train_time)
            basic_results['n_epochs'].append(basic_n_epochs)

            tf.keras.backend.clear_session()
            gc.collect()

        # Save evaluation and training history lists.
        with open(constants.basic_results_dir / 'basic_results.pkl', 'wb') as f:
            pickle.dump(basic_results, f)
    save = True
    basic_results_file = constants.basic_results_dir / f'{datetime.now().strftime("%Y%m%d")}_basic_results.txt'

    best_val_fold = evaluate.get_basic_results(basic_results, basic_results_file, save)
        
    return best_val_fold

## Phase Two: Continual Learning
def cont(
        num_basic_classes,
        emotion_list,
        emotion_ids,
        basic_emotion_list,
        basic_emotion_ids,
        epochs,
        cont_batch_size,
        cont_lr,
        file_list,
        subj_list,
        fold_length,
        img_height,
        img_width,
        cont_frozen_layers,
        es_acc_callback,
        patience,
        data_augmentation,
        excluded_emotions,
        cont_fold,
        cont_val_fold,
        cont_mem_mode,
        ):

    if (constants.basic_results_dir / 'basic_results.pkl').exists():
        with open(constants.basic_results_dir / 'basic_results.pkl', 'rb') as f:
            basic_results = pickle.load(f)
    else:
        print('Basic results not found. Please run the Basic FER Phase first.')
        return
    
    if len(excluded_emotions) > 1:
        cont_results_dir = constants.cont_results_dir / f'cont_fold_{cont_fold}_exclude_{"_".join([x.lower() for x in excluded_emotions if x.lower() != "neutral"])}'
        cont_models_dir = constants.cont_models_dir / f'cont_fold_{cont_fold}_exclude_{"_".join([x.lower() for x in excluded_emotions if x.lower() != "neutral"])}'
        cont_images_dir = constants.cont_images_dir / f'cont_fold_{cont_fold}_exclude_{"_".join([x.lower() for x in excluded_emotions if x.lower() != "neutral"])}'
    else:
        cont_results_dir = constants.cont_results_dir / f'cont_fold_{cont_fold}'
        cont_models_dir = constants.cont_models_dir / f'cont_fold_{cont_fold}'
        cont_images_dir = constants.cont_images_dir / f'cont_fold_{cont_fold}'

    if not cont_results_dir.exists():
        cont_results_dir.mkdir(parents = True)
    if not cont_models_dir.exists():
        cont_models_dir.mkdir(parents = True)
    if not cont_images_dir.exists():
        cont_images_dir.mkdir(parents = True)

    shuffled_complex_emotions = process.get_complex_emotion_lists(
        emotion_list,
        emotion_ids,
        num_basic_classes,
        excluded_emotions
        )

    complex_emotion_list = shuffled_complex_emotions['lists'][cont_fold]
    complex_emotion_ids = shuffled_complex_emotions['ids'][cont_fold]
    cont_results_file = cont_results_dir / 'cont_results.pkl'
    if cont_results_file.exists():
        with open(cont_results_file, 'rb') as f:
            cont_results = pickle.load(f)
            save = False

    else:
        save = True
        print('----Continual Learning Phase----')
        print('--------------------------------\n')
        print(f'Using complex emotion list {cont_fold}:')
        for emotion in complex_emotion_list:
            print(emotion)
        print('\n')

        cont_results = {
            'history' : [],
            'cm' : [],
            'accuracy' : [],
            'report' : [],
            'train_time' : [],
            'n_epochs' : []
        }

        new_emotion_list = basic_emotion_list[:]
        new_emotion_ids = basic_emotion_ids[:]
        num_classes = num_basic_classes

        basic_model = models.load_model(constants.basic_models_dir / f'basic_model_{cont_val_fold}')

        for layer in basic_model.layers:
            layer.trainable = False

        basic_model.compile(
            optimizer = optimizers.Adam(learning_rate = cont_lr),
            loss = losses.CategoricalCrossentropy(from_logits = True),
            metrics = ['accuracy']
        )

        train_file_list, val_file_list, basic_train_file_list, basic_val_file_list = process.generate_file_lists(
            basic_emotion_ids,
            file_list,
            subj_list,
            fold_length,
            val_fold = cont_val_fold)

        # Take 30% of available data
        M = math.floor(len(train_file_list)*3/10)
        m = math.floor(M/len(emotion_list))

        for i in range(len(complex_emotion_list)):

            print(f'--Iteration {i + 1}--\n')
            clear_session()
            gc.collect()

            print(f'Size of original training dataset: {len(train_file_list)}')
            print(f'Size of each class in original training dataset: {math.floor(len(train_file_list)/len(emotion_list))}')
            print(f'Size of Representative Memory (M): {M}')
            print(f'Size of each class in Representative Memory (m): {m}\n')

            if i == 0:
                cont_model = basic_model
                cont_train_ds = tf.data.Dataset.from_tensor_slices(basic_train_file_list).map(lambda x: process.process_path(x, basic_emotion_ids, num_classes, img_height, img_width))
                cont_val_ds = tf.data.Dataset.from_tensor_slices(basic_val_file_list).map(lambda x: process.process_path(x, basic_emotion_ids, num_classes, img_height, img_width))
                new_train_ds = None

            cont_val_ds = cont_val_ds.map(lambda x, y: (x, tf.pad(y, constants.paddings)))

            # Update representative memory.
            print(f'Updating Representative Memory...')
            cont_train_ds = train.RepresentativeMemoryUpdate(
                cont_model,
                cont_train_ds,
                new_train_ds,
                m,
                num_classes,
                num_basic_classes,
                cont_batch_size,
                )

            # Get new class info and dataset.
            new_emotion = complex_emotion_list[i]
            new_emotion_id = complex_emotion_ids[i]
            new_emotion_list.append(new_emotion)
            new_emotion_ids.append(new_emotion_id)
            num_classes = len(new_emotion_list)

            print('\n')
            print(f'Number of classes this iteration: {num_classes}')
            print(f'New class: {new_emotion}\n')

            new_train_file_list = np.array([f for f in train_file_list if os.path.split(f)[1].startswith(new_emotion_id)])
            new_val_file_list = np.array([f for f in val_file_list if os.path.split(f)[1].startswith(new_emotion_id)])

            # Construct datasets with new class.
            new_train_ds = tf.data.Dataset.from_tensor_slices(new_train_file_list).map(lambda x: process.process_path(x, new_emotion_ids, num_classes, img_height, img_width))
            cont_train_batches = cont_train_ds.concatenate(new_train_ds.shuffle(len(new_train_file_list)).take(m))\
                                              .shuffle(len(train_file_list))\
                                              .batch(cont_batch_size, drop_remainder = True)\
                                              .map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(1)

            new_val_ds = tf.data.Dataset.from_tensor_slices(new_val_file_list).map(lambda x: process.process_path(x, new_emotion_ids, num_classes, img_height, img_width))
            cont_val_ds = cont_val_ds.concatenate(new_val_ds)
            cont_val_batches = cont_val_ds.batch(cont_batch_size).prefetch(1)

            t = time.process_time()
            cont_model, cont_history, cont_cm, cont_accuracy, cont_report, cont_n_epochs = train.ContinualLearning(
                cont_model,
                basic_model,
                i,
                num_classes,
                num_basic_classes,
                img_height,
                img_width,
                cont_frozen_layers,
                cont_lr,
                cont_train_batches,
                cont_val_batches,
                epochs,
                patience,
                es_acc_callback,
                new_emotion_list,
                cont_images_dir,
        )
            cont_train_time = time.process_time() - t

            print(f'Training time: {cont_train_time}')

            if i == len(complex_emotion_list) - 1:
                cont_model.save(cont_models_dir)

            cont_results['history'].append(cont_history)
            cont_results['cm'].append(cont_cm)
            cont_results['accuracy'].append(cont_accuracy)
            cont_results['report'].append(cont_report)
            cont_results['train_time'].append(cont_train_time)
            cont_results['n_epochs'].append(cont_n_epochs)

            if cont_mem_mode == 'limit':
                # Representative Memory Limit mode (constant M, decreasing m)
                m = math.floor(M/num_classes)
            elif cont_mem_mode == 'grow':
                # Representative Memory Growth mode (increasing M, constant m)
                M += m

        with open(cont_results_file, 'wb') as f:
            pickle.dump(cont_results, f)
    save = True
    cont_results_text_file = cont_results_dir / f'{datetime.now().strftime("%Y%m%d")}_cont_results.txt'

    evaluate.get_cont_results(
        cont_results,
        basic_results,
        num_basic_classes,
        emotion_list,
        cont_images_dir,
        cont_fold,
        cont_val_fold,
        basic_emotion_list,
        complex_emotion_list,
        patience,
        cont_results_text_file,
        save
    )

def fewshot(
        num_basic_classes,
        emotion_list,
        emotion_ids,
        basic_emotion_list,
        basic_emotion_ids,
        epochs,
        fewshot_batch_size,
        cont_lr,
        file_list,
        subj_list,
        fold_length,
        img_height,
        img_width,
        cont_frozen_layers,
        es_acc_callback,
        patience,
        data_augmentation,
        cont_val_fold,
        n_shots
        ):
    
    fewshot_results_dir = constants.fewshot_results_dir / f'n_shots_{str(n_shots).lower()}'
    fewshot_models_dir = constants.fewshot_models_dir / f'n_shots_{str(n_shots).lower()}'
    fewshot_images_dir = constants.fewshot_images_dir / f'n_shots_{str(n_shots).lower()}'

    if not fewshot_results_dir.exists():
        fewshot_results_dir.mkdir(parents = True)
    if not fewshot_models_dir.exists():
        fewshot_models_dir.mkdir(parents = True)
    if not fewshot_images_dir.exists():
        fewshot_images_dir.mkdir(parents = True)

    complex_emotion_list = emotion_list[num_basic_classes:]
    complex_emotion_ids = emotion_ids[num_basic_classes:]

    if (fewshot_results_dir / 'fewshot_results.pkl').exists():
        with open(fewshot_results_dir / 'fewshot_results.pkl', 'rb') as f:
            fewshot_results = pickle.load(f)
            save = False

    else:
        save = True
        print('----Fewshot Learning Phase----')
        print('------------------------------\n')
        print(f'Initialising parameters...')

        fewshot_results = {
            'history' : [],
            'cm' : [],
            'accuracy' : [],
            'report' : [],
            'train_time' : [],
            'n_epochs' : []
        }

        basic_model = models.load_model(constants.basic_models_dir / f'basic_model_{cont_val_fold}')

        for layer in basic_model.layers:
            layer.trainable = False

        basic_model.compile(
            optimizer = optimizers.Adam(learning_rate = cont_lr),
            loss = losses.CategoricalCrossentropy(from_logits = True),
            metrics = ['accuracy']
        )

        train_file_list, val_file_list, basic_train_file_list, basic_val_file_list = process.generate_file_lists(
            basic_emotion_ids,
            file_list,
            subj_list,
            fold_length,
            cont_val_fold
            )

        print(f'Updating Representative Memory...')
        # mem_ds = RepresentativeMemoryUpdate(basic_model, basic_train_ds, m)

        for i in range(len(complex_emotion_list)):

            print(f'Fewshot Learning Phase: Iteration {i + 1}\n')

            fewshot_model = models.load_model(constants.basic_models_dir / f'basic_model_{cont_val_fold}')

            # Get new class info and dataset.
            new_emotion_list = basic_emotion_list[:]
            new_emotion_ids = basic_emotion_ids[:]

            new_emotion = complex_emotion_list[i]
            new_emotion_id = complex_emotion_ids[i]

            new_emotion_list.append(new_emotion)
            new_emotion_ids.append(new_emotion_id)

            num_classes = len(new_emotion_list)

            print(f'New class: {new_emotion}\n')

            new_train_file_list = np.array([f for f in train_file_list if os.path.split(f)[1].startswith(new_emotion_id)])
            if n_shots.lower() != 'all':
                new_train_file_list = np.random.choice(new_train_file_list, n_shots, replace=False)
            new_val_file_list = np.array([f for f in val_file_list if os.path.split(f)[1].startswith(new_emotion_id)])

            # Construct datasets with new class.
            new_train_ds = tf.data.Dataset.from_tensor_slices(new_train_file_list).map(
                lambda x: process.process_path(
                    x,
                    new_emotion_ids,
                    num_classes,
                    img_height,
                    img_width
                    )
                )
            fewshot_train_batches = new_train_ds.repeat(fewshot_batch_size).batch(fewshot_batch_size).map(lambda x, y: (data_augmentation(x, training=True), y)).prefetch(1)

            new_val_ds = tf.data.Dataset.from_tensor_slices(new_val_file_list).map(
                lambda x: process.process_path(
                    x,
                    new_emotion_ids,
                    num_classes,
                    img_height,
                    img_width
                    )
                )
            fewshot_val_batches = new_val_ds.batch(fewshot_batch_size).prefetch(1)

            t = time.process_time()
            fewshot_model, fewshot_history, fewshot_cm, fewshot_accuracy, fewshot_report, fewshot_n_epochs = train.fewshot_learning(
                fewshot_model,
                basic_model,
                i,
                num_classes,
                num_basic_classes,
                img_height,
                img_width,
                cont_frozen_layers,
                cont_lr,
                fewshot_train_batches,
                fewshot_val_batches,
                epochs,
                patience,
                es_acc_callback,
                new_emotion_list,
                fewshot_images_dir
                )
            fewshot_train_time = time.process_time() - t

            print(f'Training time: {fewshot_train_time}')

            fewshot_model.save(fewshot_models_dir / f'fewshot_model_{i}')

            fewshot_results['history'].append(fewshot_history)
            fewshot_results['cm'].append(fewshot_cm)
            fewshot_results['accuracy'].append(fewshot_accuracy)
            fewshot_results['report'].append(fewshot_report)
            fewshot_results['train_time'].append(fewshot_train_time)
            fewshot_results['n_epochs'].append(fewshot_n_epochs)

            del fewshot_model

            clear_session()
            gc.collect()

        with open(fewshot_results_dir / f'fewshot_results.pkl', 'wb') as f:
            pickle.dump(fewshot_results, f)
    save = True
    fewshot_results_file = fewshot_results_dir / f'{datetime.now().strftime("%Y%m%d")}_fewshot_results.txt'
    evaluate.get_fewshot_results(fewshot_results, fewshot_results_file, complex_emotion_list, save)

def vis(
        emotion_list,
        emotion_ids,
        basic_emotion_list,
        basic_emotion_ids,
        subj_list,
        file_list,
        num_basic_classes,
        img_height,
        img_width,
        cont_val_fold,
        subj
        ):
    print('----Visualisation----')
    print('---------------------\n')

    basic_model = models.load_model(constants.basic_models_dir / f'basic_model_{cont_val_fold}')

    subj_file_list = visualise.get_subj_file_list(
        subj,
        emotion_ids,
        subj_list,
        file_list
    )

    complex_emotion_list = emotion_list[num_basic_classes:]
    complex_emotion_ids = emotion_ids[num_basic_classes:]

    visualise.plot_temperatures(
        subj,
        basic_model,
        basic_emotion_list,
        basic_emotion_ids,
        subj_file_list,
        num_basic_classes,
        img_height,
        img_width)

    visualise.get_basic_CAMs(
        basic_model,
        basic_emotion_list,
        basic_emotion_ids,
        subj_file_list,
        num_basic_classes,
        img_height,
        img_width
        )
    
    visualise.get_complex_CAMs(
        complex_emotion_list,
        complex_emotion_ids,
        basic_emotion_ids,
        subj_file_list,
        num_basic_classes,
        img_height,
        img_width
    )
    
if __name__=='__main__':
    main()