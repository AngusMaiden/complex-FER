#!/usr/bin/env python
# coding: utf-8


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import math
from tabulate import tabulate
from glob import glob
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
import random
from copy import deepcopy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, initializers
from tensorflow.keras.utils import to_categorical, plot_model, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from deepface import DeepFace
from deepface.commons.functions import normalize_input
import keras_tuner as kt



set_random_seed(42)



gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
       #virtual devices must be set before GPUs have been initialized
        print(e)


# ## Data Preprocessing


def decode_label(file_path):
    
    idx = os.path.split(file_path.numpy().decode('UTF-8'))[1][:2]
    
    # Get numerical emotion class and convert to categorical.
    label = to_categorical(emotion_ids.index(idx), num_classes = num_classes)
    
    return label
  
def detect_face(file_path):
    
    # RetinaFace face detection.
    img = DeepFace.detectFace(img_path = file_path.numpy().decode('UTF-8'),
                              target_size = (img_height, img_width),
                              detector_backend = 'retinaface',
                              enforce_detection = False,
                              align = False)
    
    # De-normalise image after DeepFace normalisation (we will use a different normalisation aligned to our pre-trained model).
    img = normalize_input(img, 'raw')
    
    return img

def process_path(file_path):
    
    # Get label from file path.
    label = tf.py_function(decode_label, [file_path], tf.int32)
    label.set_shape(tf.TensorShape([num_classes]))
    
    # Get image from file path using DeepFace face detection library.
    img = tf.py_function(detect_face, [file_path], tf.float32)
    
    # Normalisation aligned to that used by pre-trained model.
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    img.set_shape(tf.TensorShape([img_height, img_width, 3]))
    
    return img, label



root_dir = os.getcwd()
data_dir = 'CFEE_Database_230'
models_dir = 'Models/Current'
checkpoints_dir = 'Models/Checkpoints'
results_dir = 'Results/Current'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

img_height = 299
img_width = 299
num_basic_classes = 6

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

es_loss_callback = EarlyStopping(monitor = 'val_loss', patience = 100, restore_best_weights = True)
es_acc_callback = EarlyStopping(monitor = 'val_accuracy', patience = 100, restore_best_weights = True)

cl_initializer = initializers.GlorotUniform()

data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"),
                                      layers.RandomRotation(0.01),
                                      layers.RandomZoom(0.05),
                                      layers.RandomTranslation(0.05, 0.05)], name = 'data_augmentation')

if os.path.isfile('annotation_dict.pkl'):
    with open('annotation_dict.pkl', 'rb') as f:
        annotation_dict = pickle.load(f)
else:
    annotation_dict = pd.read_excel(f'{data_dir}/AU_annotation_all_subjects.xls', sheet_name = None)
    for k, v in annotation_dict.items():
        annotation_dict[k] = v.fillna(0)
    with open('annotation_dict.pkl', 'wb') as f:
        pickle.dump(annotation_dict, f)

emotion_list = annotation_dict['summary'].iloc[:,1].to_list()[1:]
emotion_ids = annotation_dict['summary'].iloc[:,0].to_list()[1:]
emotion_ids = [f'{emotion_id:02d}' for emotion_id in emotion_ids]



file_list = [f for f in glob(os.path.join(data_dir, '*/Images*/*.[jJ][pP][gG]')) \
             if any([idx == os.path.split(f)[1][:2] for idx in emotion_ids])]
idx_list = sorted(list(set([os.path.split(f)[1][:2] for f in file_list])))
subj_list = sorted(list(set([os.path.split(f)[1][3:6] for f in file_list])))

fold_length = int(len(subj_list)/10)



def evaluate_model(model, test_batches, class_list):
    
    test_logits = model.predict(test_batches)
    test_preds = np.argmax(test_logits, axis = -1)
    test_labels = np.concatenate([y for x, y in test_batches])
    test_labels = np.argmax(test_labels, axis = -1)
    
    cm = confusion_matrix(test_labels, test_preds)
    
    cm_disp = ConfusionMatrixDisplay(cm, display_labels = class_list)
    fig, ax = plt.subplots(figsize = (6,6))
    cm_disp.plot(ax = ax, xticks_rotation = 'vertical')
    plt.suptitle(f'Confusion Matrix', fontsize = 10)
    plt.show()
    
    accuracy = accuracy_score(test_labels, test_preds)
    
    print(f'\nAccuracy: {accuracy*100:.2f}%')
    
    report = classification_report(test_labels, test_preds, target_names = class_list)
    print(report)
    
    return cm, accuracy, report



def cross_validation(test_fold = 0, val_fold = 0):
    val_file_list = []
    test_file_list = []
    
    test_subj_list = subj_list[test_fold*fold_length:fold_length+test_fold*fold_length]
    val_subj_list = [subj for subj in subj_list if subj not in test_subj_list][val_fold*fold_length:fold_length+val_fold*fold_length]
    
    for subj in test_subj_list:
        test_file_list.extend([f for f in file_list if os.path.split(f)[1][3:6] == subj])
    
    for subj in val_subj_list:
        val_file_list.extend([f for f in file_list if os.path.split(f)[1][3:6] == subj])
    
    train_file_list = [f for f in file_list if f not in val_file_list and f not in test_file_list]
    
    return sorted(train_file_list), sorted(val_file_list), sorted(test_file_list)



def generate_file_lists(test_fold = 0, val_fold = 0):

    train_file_list, val_file_list, test_file_list = cross_validation(test_fold, val_fold)

    basic_train_file_list = []
    for i in basic_emotion_ids:
        for f in train_file_list:
            if str(os.path.split(f)[1]).startswith(i):
                basic_train_file_list.append(f)
    
    basic_val_file_list = []
    for i in basic_emotion_ids:
        for f in val_file_list:
            if str(os.path.split(f)[1]).startswith(i):
                basic_val_file_list.append(f)
    
    basic_test_file_list = []
    for i in basic_emotion_ids:
        for f in test_file_list:
            if str(os.path.split(f)[1]).startswith(i):
                basic_test_file_list.append(f)
                
    return train_file_list, val_file_list, test_file_list, basic_train_file_list, basic_val_file_list, basic_test_file_list


# ### Hyperparameter tuning functions


def build_FER_model_tuned(
    num_classes,
    units_1,
    units_2,
    activation_1,
    activation_2,
    regularizer_1,
    regularizer_2,
    dropout1,
    dropout2,
    dropout3,
    lr,
    epsilon,
    global_pooling,
    #base_model
    ):
    
    base_model = ResNet50V2(weights = 'imagenet', input_shape=(img_height, img_width, 3), include_top = False, pooling = global_pooling)
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = layers.Dropout(dropout1)(x)
    x = layers.Dense(units_1, activation = activation_1, kernel_regularizer = l2(regularizer_1))(x)
    x = layers.Dropout(dropout2)(x)
    x = layers.Dense(units_2, activation = activation_2, kernel_regularizer = l2(regularizer_2))(x)
    feature_output = layers.Dropout(dropout3)(x)
    classifier_output = layers.Dense(num_classes)(feature_output)
    
    model = keras.Model(base_model.input, classifier_output)
    
    model.compile(optimizer = optimizers.Adam(learning_rate = lr, epsilon = epsilon),
                  loss = losses.CategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model



def FER_tuning_model(hp):
    
    units_1 = hp.Choice("units_1", [32, 64, 128, 256, 512, 1024], default = 1024)
    units_2 = hp.Choice("units_2", [32, 64, 128, 256, 512, 1024], default = 512)
    
    regularizer_1 = hp.Choice("regularizer_1", [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6], default = 1e-4)
    regularizer_2 = hp.Choice("regularizer_2", [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6], default = 1e-4)
    
    activation_1 = hp.Choice("activation_1", ['relu', 'sigmoid'], default = 'sigmoid')
    activation_2 = hp.Choice("activation_2", ['relu', 'sigmoid'], default = 'sigmoid')
    
    dropout1 = hp.Choice("dropout1", [0.0, 0.1, 0.2, 0.3, 0.5], default = 0.2)
    
    dropout2 = hp.Choice("dropout2", [0.0, 0.1, 0.2, 0.3, 0.5], default = 0.2)
    
    dropout3 = hp.Choice("dropout3", [0.0, 0.1, 0.2, 0.3, 0.5], default = 0.2)
    
    lr = hp.Choice("lr", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6], default = 1e-4)
    epsilon = hp.Choice("epsilon", [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6], default = 0.1)
    
    global_pooling = hp.Choice("global_pooling", ['max', 'avg'], default = 'avg')
    
    #base_model_type = hp.Choice("base_model_type", ["xception", "inception", "resnet"], default = "resnet")
    
    #if base_model_type == "xception":
    #    with hp.conditional_scope("base_model_type", "xception"):
    #        base_model = Xception(weights = 'imagenet', input_shape=(img_height, img_width, 3), include_top = False, pooling = global_pooling)
    
    #if base_model_type == "inception":
    #    with hp.conditional_scope("base_model_type", "inception"):
    #        base_model = InceptionV3(weights = 'imagenet', input_shape=(img_height, img_width, 3), include_top = False, pooling = global_pooling)
    
    #if base_model_type == "resnet":
    #    with hp.conditional_scope("base_model_type", "resnet"):
    #        base_model = ResNet50V2(weights = 'imagenet', input_shape=(img_height, img_width, 3), include_top = False, pooling = global_pooling)
    
    model = build_FER_model_tuned(
        num_classes,
        units_1,
        units_2,
        activation_1,
        activation_2,
        regularizer_1,
        regularizer_2,
        dropout1,
        dropout2,
        dropout3,
        lr,
        epsilon,
        global_pooling,
        #base_model
        )
    
    return model



val_fold, test_fold = 3, 0
num_classes = num_basic_classes
basic_emotion_list = emotion_list[:num_classes]
basic_emotion_ids = emotion_ids[:num_classes]



# Generate files lists using the current validation fold.
train_file_list, val_file_list, test_file_list, basic_train_file_list, basic_val_file_list, basic_test_file_list \
= generate_file_lists(test_fold, val_fold)

# Generate datasets using the current validation fold.
basic_train_ds = tf.data.Dataset.from_tensor_slices(basic_train_file_list).map(process_path).cache()
basic_train_batches = basic_train_ds.shuffle(len(basic_train_ds), reshuffle_each_iteration=True)\
                                    .batch(batch_size)\
                                    .map(lambda x, y: (data_augmentation(x, training = True), y))\
                                    .prefetch(AUTOTUNE)

basic_val_batches = tf.data.Dataset.from_tensor_slices(basic_val_file_list).map(process_path)\
                                                                           .batch(batch_size)\
                                                                           .cache()\
                                                                           .prefetch(AUTOTUNE)

basic_test_batches = tf.data.Dataset.from_tensor_slices(basic_test_file_list).map(process_path)\
                                                                             .batch(batch_size)\
                                                                             .cache()\
                                                                             .prefetch(AUTOTUNE)

FER_tuning_model(kt.HyperParameters())
FER_tuner = kt.Hyperband(hypermodel = FER_tuning_model,
                         objective = "val_accuracy",
                         max_epochs = 350,
                         overwrite = True,
                         directory = r'D:\KerasTuner',
                         project_name = "FER_tuning",
                         hyperband_iterations = 1)

FER_tuner.search(basic_train_batches,
                 validation_data = basic_val_batches,
                 epochs = 350,
                 callbacks = [es_acc_callback])
FER_tuner.results_summary()
best_tuned_FER_models = FER_tuner.get_best_models(num_models = 1)
FER_tuned_model = best_tuned_FER_models[0]
FER_tuned_model.save(os.path.join(models_dir, f'FER_tuned_model_{val_fold}_{test_fold}'))

FER_tuned_cm, FER_tuned_accuracy, FER_tuned_report = evaluate_model(FER_tuned_model, basic_test_batches, basic_emotion_list)

