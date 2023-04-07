import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from matplotlib import pyplot as plt
import gc
import cv2

from .. import constants
from ..data import process

def get_image_and_label(file, emotion_list, emotion_ids, num_classes, img_height, img_width):
    img, label = process.process_path(file, emotion_ids, num_classes, img_height, img_width)
    img = (img.numpy() * 255).astype('uint8')
    label = emotion_list[np.argmax(label)]
    return img, label

def get_subj_file_list(subj, emotion_ids, subj_list, file_list):
    subj_id = subj_list[subj]
    subj_file_list = []
    for emotion_id in emotion_ids:
        for file in file_list:
            if os.path.split(file)[1][:2] == emotion_id and os.path.split(file)[1][3:6] == subj_id:
                subj_file_list.append(file)

    return subj_file_list

def plot_temperatures(subj, model, emotion_list, emotion_ids, subj_file_list, num_classes, img_height, img_width):

    temperatures = [1, 2, 5, 10]

    logits_list = []
    
    plt.figure(figsize=(15, 5))
    for i, T in enumerate(temperatures):
        img, label = process.process_path(subj_file_list[subj], emotion_ids, num_classes, img_height, img_width)
        label = emotion_list[np.argmax(label)]
        
        T_layer = layers.Lambda(lambda x:x/T)(model.output, training = False)

        # Create a softmax layer
        dist_layer = layers.Softmax()(T_layer)
        
        # Add the model t_layer to the whole model
        dist_model = Model(model.input, dist_layer)
        
        # Append for plotting
        logits_list.append(dist_model.predict(tf.expand_dims(img, axis = 0)))
    
        plt.plot(logits_list[i][0], label = str(T))
        plt.xticks(np.arange(num_classes), emotion_list, fontsize=16, rotation='vertical')
    
    #plt.title(f'Different temperatures applied to Output Layer of "{label}" image (FER Model).')
    plt.legend(temperatures, title = 'Temperatures')
    plt.savefig(constants.images_dir / 'visualisations' / 'softmax_output_with_T.jpg', bbox_inches = 'tight', dpi = 1000)
    try:
        get_ipython
        plt.show()
    except NameError:
        plt.clf()

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_layer()

    def find_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find layer, cannot apply CAM.")

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return heatmap, output

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        
        return heatmap

def CAM(model, img_file, ids, idx, out_file, img_height, img_width, num_classes, emotion):
    img, label = process.process_path(img_file, ids, num_classes, img_height, img_width)
    
    orig_img = (img.numpy() * 255).astype('uint8')
    
    img = tf.expand_dims(img, axis = 0)
    
    cam = GradCAM(model, idx)
    heatmap = cam.compute_heatmap(img)
    heatmap, output = cam.overlay_heatmap(heatmap, orig_img, 0.5, cv2.COLORMAP_HOT)
    
    #show_image(heatmap, orig_label)
    plt.imshow(output)
    plt.title(emotion, fontsize=20)
    plt.axis('off')
    plt.savefig(out_file, bbox_inches = 'tight', dpi = 1000)
    try:
        get_ipython
        plt.show()
    except NameError:
        plt.clf()

def get_basic_CAMs(model, basic_emotion_list, basic_emotion_ids, subj_file_list, num_basic_classes, img_height, img_width):
    num_classes = num_basic_classes
    tf.keras.backend.clear_session()

    for i, emotion in enumerate(basic_emotion_list):
        CAM(
            model,
            subj_file_list[i],
            basic_emotion_ids,
            i,
            constants.images_dir / 'visualisations' / f'gradCAM_{emotion}.jpg',
            img_height,
            img_width,
            num_classes,
            emotion
        )
        tf.keras.backend.clear_session()
        gc.collect()

def get_complex_CAMs(complex_emotion_list, complex_emotion_ids, basic_emotion_ids, subj_file_list, num_basic_classes, img_height, img_width):
    num_classes = num_basic_classes + 1
    tf.keras.backend.clear_session()

    for i, emotion in enumerate(complex_emotion_list):
        fewshot_emotion_ids = basic_emotion_ids+[complex_emotion_ids[i]]

        model = models.load_model(constants.fewshot_models_dir / 'n_shots_all' / f'fewshot_model_{i}')

        CAM(
            model,
            subj_file_list[i+num_basic_classes],
            fewshot_emotion_ids,
            num_basic_classes,
            constants.images_dir / 'visualisations' / f'gradCAM_{emotion}.jpg',
            img_height,
            img_width,
            num_classes,
            emotion
        )
        tf.keras.backend.clear_session()
        gc.collect()