import os
from numpy import array as np_array
from sklearn.model_selection import StratifiedKFold
from tensorflow import py_function, TensorShape, float32
from tensorflow.io import read_file, decode_jpeg
from tensorflow.image import convert_image_dtype
from tensorflow.keras.utils import to_categorical
from PIL import Image
from glob import glob
from deepface import DeepFace
import random
import pickle

from .. import constants

def detectface(emotion_ids, data_dir):
    file_list = [f for f in data_dir.glob('*/Images*/*[!_detectface].[jJ][pP][gG]')\
                 if any([idx == os.path.split(f)[1][:2] for idx in emotion_ids])]
    
    for filename in file_list:
        new_filename = (
            constants.processed_data_dir /
            filename.parents[2].name /
            filename.parents[1].name /
            filename.parents[0].name /
            (filename.stem + '_detectface' + filename.suffix)
        )
        
        if not new_filename.exists():
            img = DeepFace.detectFace(img_path = filename,
                                      target_size = (constants.img_height, constants.img_width),
                                      detector_backend = 'retinaface',
                                      enforce_detection = False,
                                      align = False)
            
            img = Image.fromarray((img * 255).astype('uint8')).convert('RGB')
            
            img.save(new_filename)

def decode_label(filename, ids, num_classes):
    idx = os.path.split(filename.numpy().decode('UTF-8'))[-1][:2]
    ids = [i.decode('utf-8') for i in ids.numpy().tolist()]
    label = to_categorical(ids.index(idx), num_classes = num_classes)
    
    return label

def parse_image(filename):
    img = read_file(filename)
    img = decode_jpeg(img)
    img = convert_image_dtype(img, float32)
    
    return img

def process_path(filename, ids, num_classes, img_height, img_width):
    label = py_function(decode_label, [filename, ids, num_classes], float32)
    label.set_shape(TensorShape([num_classes]))
    img = py_function(parse_image, [filename], float32)
    img.set_shape(TensorShape([img_height, img_width, 3]))
    
    return img, label

def cross_validation(file_list, subj_list, fold_length, val_fold = 0):
    val_file_list = []
    
    val_subj_list = subj_list[val_fold*fold_length:fold_length+val_fold*fold_length]
    
    for subj in val_subj_list:
        val_file_list.extend([f for f in file_list if os.path.split(f)[1][3:6] == subj])
    
    train_file_list = [f for f in file_list if f not in val_file_list]
    
    return sorted(train_file_list), sorted(val_file_list)

def generate_file_lists(basic_emotion_ids, file_list, subj_list, fold_length, val_fold = 0):

    train_file_list, val_file_list = cross_validation(file_list, subj_list, fold_length, val_fold)

    basic_train_file_list = [f for f in train_file_list if os.path.split(f)[1][:2] in basic_emotion_ids]
    basic_val_file_list = [f for f in val_file_list if os.path.split(f)[1][:2] in basic_emotion_ids]
    
    train_file_list = np_array(train_file_list)
    val_file_list = np_array(val_file_list)
    basic_train_file_list = np_array(basic_train_file_list)
    basic_val_file_list = np_array(basic_val_file_list)
    
    return train_file_list, val_file_list, basic_train_file_list, basic_val_file_list

def get_complex_emotion_lists(emotion_list, emotion_ids, num_basic_classes, excluded_emotions):    
    if len(excluded_emotions) > 1:
        shuffled_complex_emotions_file = (
            constants.interim_data_dir /
            f'shuffled_complex_emotions_exclude_{"_".join([x.lower() for x in excluded_emotions if x.lower() != "neutral"])}.pkl'
        )
    else:
        shuffled_complex_emotions_file = constants.interim_data_dir / 'shuffled_complex_emotions.pkl'
    if shuffled_complex_emotions_file.exists():
        with open(shuffled_complex_emotions_file, 'rb') as f:
            shuffled_complex_emotions = pickle.load(f)
    else:
        complex_emotion_list = emotion_list[num_basic_classes:]
        complex_emotion_ids = emotion_ids[num_basic_classes:]

        shuffled_complex_emotions = { 'lists' : [], 'ids' : [] }

        for _ in range(10):
            shuffled_list = complex_emotion_list[:]
            shuffled_ids = complex_emotion_ids[:]
            shuffled_zip = [list(i) for i in zip(shuffled_list, shuffled_ids)]

            random.shuffle(shuffled_zip)
            shuffled_list, shuffled_ids = [list(i) for i in zip(*shuffled_zip)]

            shuffled_complex_emotions['lists'].append(shuffled_list)
            shuffled_complex_emotions['ids'].append(shuffled_ids)

        with open(shuffled_complex_emotions_file, 'wb') as f:
            pickle.dump(shuffled_complex_emotions, f)
    
    return shuffled_complex_emotions