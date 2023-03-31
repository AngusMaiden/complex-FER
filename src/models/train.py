import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import layers, models, optimizers, losses, initializers, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.regularizers import l2
from tensorflow.data import Dataset

import constants
from . import evaluate

cl_initializer = initializers.GlorotUniform()

def build_basic_model(num_classes, img_height, img_width, lr):
    
    base_model = ResNet50V2(weights = 'imagenet',
                            input_shape=(img_height, img_width, 3),
                            include_top = False,
                            pooling = 'max')
    base_model._name = 'ResNet_FER'
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        512,
        activation = 'sigmoid',
        kernel_regularizer = l2(1e-5)
    )(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        32,
        activation = 'sigmoid',
        kernel_regularizer = l2(1e-5)
    )(x)
    x = layers.Dropout(0.2)(x)
    classifier_output = layers.Dense(num_classes)(x)
    
    model = Model(base_model.input, classifier_output)
    
    model.compile(optimizer = optimizers.Adam(learning_rate = lr),
                  loss = losses.CategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy'])
    
    return model

def train_basic_model(model, basic_train_ds, basic_val_ds, epochs, es_callback, basic_frozen_layers, finetuning_lr, save_model_dir):
    
    basic_history = model.fit(basic_train_ds,
                              validation_data = basic_val_ds,
                              epochs = epochs,
                              callbacks = [es_callback])
    
    basic_history = basic_history.history
    
    for layer in model.layers[basic_frozen_layers:]:
        if not layer.__class__.__name__ == 'BatchNormalization':
            layer.trainable = True
    
    model.compile(optimizer = optimizers.Adam(learning_rate = finetuning_lr),
                      loss = losses.CategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])
    
    basic_finetuning_history = model.fit(basic_train_ds,
                                           validation_data = basic_val_ds,
                                           epochs = epochs,
                                           callbacks = [es_callback])
    
    basic_finetuning_history = basic_finetuning_history.history
    
    model.save(save_model_dir)
    
    return model, basic_history, basic_finetuning_history

def RepresentativeMemoryUpdate(model, old_ds, new_ds, m, num_classes, num_basic_classes, batch_size):
    
    # Predictive sorting method: lists samples in order of prediction probability, then selects top m samples.
    def predictive_sort(model, ds, m, idx):
        # Get images, labels from dataset.
        imgs = np.array([x for x, y in ds])
        labels = np.array([y for x, y in ds])
        
        # Generate predictions from training samples in subset
        preds = model.predict(ds.batch(batch_size))
            
        # Create sorted index of predictions
        index_sorted = np.argsort(preds[:,idx])[::-1]
        
        # Order samples by highest prediction probability
        imgs = imgs[index_sorted]
        labels = labels[index_sorted]
            
        # Take top m samples.
        imgs = imgs[:m, :]
        labels = labels[:m, :]
        
        return imgs, labels
    
    # Pad dataset with 0s for new class.
    old_ds = old_ds.map(lambda x, y: (x, tf.pad(y, paddings)))
    
    # Select m samples from each basic emotion class using predictive sorting method, if first iteration in Phase 2.
    if num_classes == num_basic_classes:
        for i in range(num_classes):
            ds = old_ds.filter(lambda x, y: y[i] == 1)
            
            # Select samples using predictive sorting method
            imgs, labels = predictive_sort(model, ds, m, i)
            
            if i == 0:
                mem_ds = Dataset.from_tensor_slices((imgs, labels))
            else:
                mem_ds = mem_ds.concatenate(Dataset.from_tensor_slices((imgs, labels)))
    else:
        for i in range(num_classes - 1):        
            ds = old_ds.filter(lambda x, y: y[i] == 1)
            
            # Get images, labels from dataset.
            imgs = np.array([x for x, y in ds])
            labels = np.array([y for x, y in ds])
                
            # Take top m samples.
            imgs = imgs[:m, :]
            labels = labels[:m, :]
            
            if i == 0:
                mem_ds = Dataset.from_tensor_slices((imgs, labels))
            else:
                mem_ds = mem_ds.concatenate(Dataset.from_tensor_slices((imgs, labels)))
    
        # Pad dataset with 0s for new class.
        new_ds = new_ds.map(lambda x, y: (x, tf.pad(y, constants.paddings)))
    
        # Select m samples from new emotion class using predictive sorting method
        new_imgs, new_labels = predictive_sort(model, new_ds, m, num_classes-1)
        mem_ds = mem_ds.concatenate(Dataset.from_tensor_slices((new_imgs, new_labels)))
    
    return mem_ds

class ContinualLearner(Model):
    def __init__(self,
                 model,
                 teacher_model,
                 i,
                 num_classes,
                 num_basic_classes,
                 img_height,
                 img_width,
                 cont_frozen_layers,
                 cont_lr
                 ):
        super(ContinualLearner, self).__init__()

        self.i = i
        self.lr = cont_lr
        
        if self.i == 0:
            # Create a new model using the Basic FER model architecture.
            self.old_model = build_basic_model(num_basic_classes, img_height, img_width, cont_lr)
            self.old_model.set_weights(model.get_weights())
                    
        else:
            # Reuse model from previous continual learning iteration.
            self.old_model = models.clone_model(model)
            self.old_model.set_weights(model.get_weights())
            
        for layer in self.old_model.layers:
                layer.trainable = False
            
        for layer in self.old_model.layers[cont_frozen_layers:]:
            if not layer.__class__.__name__ == 'BatchNormalization':
                layer.trainable = True
        
        self.old_model.compile(
                optimizer = optimizers.Adam(learning_rate = cont_lr),
                loss = losses.CategoricalCrossentropy(from_logits = True),
                metrics = ['accuracy']
            )
        
        # Define model input.
        self.img_input = self.old_model.input
        
        # Create new classification layer and get feature output layer.
        self.fe = self.old_model.layers[-2].output
        self.new_cl = layers.Dense(num_classes, name = 'cont_Dense')(self.fe)
        
        # Construct new model.
        self.new_model = Model(self.img_input, self.new_cl)
        
        # Get weights of final classification layer of old model.
        cl_weights = self.old_model.layers[-1].get_weights()
                
        # Add a node to these weights (initialised with random values for the kernel and 0s for the bias weights).
        cl_weights[1] = np.concatenate((cl_weights[1], np.zeros(1)), axis = 0, dtype = np.float32)
        cl_weights[0] = np.concatenate((cl_weights[0],
                                        cl_initializer(shape=(cl_weights[0].shape[0], 1))),
                                        axis = 1,
                                        dtype = np.float32)
        
        # Transfer final classification layer weights from the old model.
        self.new_model.layers[-1].set_weights(cl_weights)
        
        self.teacher_model = teacher_model
    
    def call(self, inputs, **kwargs):
        return self.new_model(inputs, **kwargs)
    
    def distillation_loss(self, y_true, y_pred):
        n_zeros = y_pred.shape[-1] - y_true.shape[-1]
        loss_paddings = tf.constant([[0,0],[0, n_zeros]])
        y_true = tf.pad(y_true, loss_paddings)
        dist_loss = self.dist_loss_fn(tf.nn.softmax(y_true / self.T, axis = 1),
                                      tf.nn.softmax(y_pred / self.T, axis = 1))
        return dist_loss
    
    def compile(self,
                gamma = 0.1,
                T = 3):
        
        super(ContinualLearner, self).compile(optimizer=optimizers.Adam(learning_rate = self.lr), metrics=['accuracy'])
        self.loss_fn = losses.CategoricalCrossentropy(from_logits = True)
        self.dist_loss_fn = losses.CategoricalCrossentropy(from_logits = False)
        self.gamma = gamma * np.exp(-1*(self.i / (1 + math.e)))
        self.T = T
    
    @tf.function
    def train_step(self, inputs):
 
        x_batch, y_batch = inputs
        teacher_logits = self.teacher_model(x_batch, training = False)
        
        with tf.GradientTape() as tape:
            
            #student_logits = self.student_model(x_batch, training = True)
            logits = self.new_model(x_batch, training = True)
            
            cl_loss = self.loss_fn(y_batch, logits)
            dist_loss = self.distillation_loss(teacher_logits, logits)
            loss = (1 - self.gamma) * cl_loss + self.gamma * dist_loss
        
        # Compute gradients
        trainable_vars = self.new_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y_batch, logits)
        
        # Return a dict of performance
        results = {'loss': loss, 'cl_loss' : cl_loss, 'dist_loss' : dist_loss}
        results.update({m.name: m.result() for m in self.metrics})
        
        return results
    
    @tf.function
    def test_step(self, inputs):
                
        x_batch, y_batch = inputs
        
        logits = self.new_model(x_batch, training = False)
        loss = self.loss_fn(y_batch, logits)
        
        self.compiled_metrics.update_state(y_batch, logits)
        
        results = {'loss': loss}
        results.update({m.name: m.result() for m in self.metrics})
        
        return results

def ContinualLearning(
        cont_model,
        teacher_model,
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
        es_callback,
        new_emotion_list,
        cont_images_dir,
        ):
    
    cont_model = ContinualLearner(
        cont_model,
        teacher_model,
        i,
        num_classes,
        num_basic_classes,
        img_height,
        img_width,
        cont_frozen_layers,
        cont_lr,
        )
    
    cont_model.compile()
       
    print(f'Fitting model to training data...\n')
    cont_history = cont_model.fit(
        cont_train_batches,
        validation_data = cont_val_batches,
        epochs = epochs,
        callbacks = [es_callback]
    )
    
    cont_history = cont_history.history
    
    cont_n_epochs = len(cont_history['loss'])-patience
                   
    print(f'Evaluating model...\n')
    # Plot Continual Learning Training History
    evaluate.plot_training_history(
        cont_history,
        filename = cont_images_dir / f'cont_training_history_{i}.jpg',
        patience = patience
    )
        
    # Evaluate Continual Learning Model
    cont_cm, cont_accuracy, cont_report = evaluate.evaluate_model(
        cont_model,
        cont_val_batches,
        new_emotion_list,
        cont_images_dir / f'cont_cm_{i}.jpg'
        )
    
    return (cont_model.new_model,
            cont_history,
            cont_cm,
            cont_accuracy,
            cont_report,
            cont_n_epochs)

def fewshot_learning(
    fewshot_model,
    teacher_model,
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
    es_callback,
    new_emotion_list,
    fewshot_images_dir
):
    
    fewshot_model = ContinualLearner(
        fewshot_model,
        teacher_model,
        0,
        num_classes,
        num_basic_classes,
        img_height,
        img_width,
        cont_frozen_layers,
        cont_lr
        )
    fewshot_model.compile()
       
    print(f'Fitting model to training data...\n')
    fewshot_history = fewshot_model.fit(
        fewshot_train_batches,
        validation_data = fewshot_val_batches,
        epochs = epochs,
        callbacks = [es_callback]
    )
    
    fewshot_history = fewshot_history.history
    
    fewshot_n_epochs = len(fewshot_history['loss'])-patience
        
    # Evaluate Fewshot Learning Model
    print(f'Evaluating model...\n')
    # Plot Fewshot Learning Training History
    evaluate.plot_training_history(
        fewshot_history,
        filename = fewshot_images_dir / f'fewshot_training_history_{i}.jpg',
        patience = patience
    )
    
    fewshot_cm, fewshot_accuracy, fewshot_report = evaluate.evaluate_model(
        fewshot_model,
        fewshot_val_batches,
        new_emotion_list,
        fewshot_images_dir / f'fewshot_cm_{i}.jpg'
    )
    
    return (
        fewshot_model.new_model,
        fewshot_history,
        fewshot_cm,
        fewshot_accuracy,
        fewshot_report,
        fewshot_n_epochs
    )