import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Layer, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class KANConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', kernel_regularizer=None, **kwargs):
        super(KANConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer='he_normal',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)
        self.bias = self.add_weight(shape=(self.filters,),
                                    initializer='zeros',
                                    name='bias')
        self.control_points = self.add_weight(shape=self.kernel_shape,
                                              initializer='he_normal',
                                              name='control_points')

    def call(self, inputs):
        conv_out = tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
        kan_out = self.kernel_adaptive_network(inputs)
        return conv_out + kan_out + self.bias

    def kernel_adaptive_network(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper()
        )
        
        input_shape = tf.shape(inputs)
        patches_shape = tf.shape(patches)
        patch_size = self.kernel_size[0] * self.kernel_size[1] * input_shape[-1]
        
        patches_reshaped = tf.reshape(patches, [-1, patch_size])
        control_points_reshaped = tf.reshape(self.control_points, [-1, self.filters])
        
        distances = tf.reduce_sum(tf.square(tf.expand_dims(patches_reshaped, 2) - tf.expand_dims(control_points_reshaped, 0)), axis=1)
        
        gamma = 1.0 / (2.0 * tf.reduce_mean(distances))
        kernel_output = tf.exp(-gamma * distances)
        
        kernel_output_reshaped = tf.reshape(kernel_output, [patches_shape[0], patches_shape[1], patches_shape[2], self.filters])
        
        return kernel_output_reshaped

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)
    
    def get_config(self):
        config = super(KANConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'kernel_regularizer': self.kernel_regularizer,
        })
        return config

original_data_dir = '/content/drive/MyDrive/gtprac/original_grape_data'
train_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_train'
test_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary'
)

class EarlyStoppingByMetric(Callback):
    def __init__(self, monitor1='accuracy', monitor2='recall', monitor3='recall_esca', value1=0.9, value2=0.8, value3=0.8, val_loss_threshold=0.8, verbose=1):
        super(Callback, self).__init__()
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.monitor3 = monitor3
        self.value1 = value1
        self.value2 = value2
        self.value3 = value3
        self.val_loss_threshold = val_loss_threshold
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current1 = logs.get(self.monitor1)
        current2 = logs.get(self.monitor2)
        current3 = logs.get(self.monitor3)
        current_val_loss = logs.get('val_loss')
        if current1 is None or current2 is None or current3 is None or current_val_loss is None:
            return
        
        if current1 >= self.value1 and current2 >= self.value2 and current3 >= self.value3 and current_val_loss < self.val_loss_threshold:
            if self.verbose > 0:
                print(f"Epoch {epoch+1}: early stopping threshold reached - {self.monitor1}: {current1}, {self.monitor2}: {current2}, {self.monitor3}: {current3}, val_loss: {current_val_loss}")
            self.model.stop_training = True

class RecallForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id, name='recall_for_class', **kwargs):
        super(RecallForClass, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.positives = self.add_weight(name='pos', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32)
        
        class_id_true = tf.equal(y_true, self.class_id)
        class_id_pred = tf.equal(y_pred, self.class_id)

        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(class_id_true, class_id_pred), self.dtype))
        positives = tf.reduce_sum(tf.cast(class_id_true, self.dtype))

        self.true_positives.assign_add(true_positives)
        self.positives.assign_add(positives)

    def result(self):
        return self.true_positives / (self.positives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.positives.assign(0)

class ModelCheckpointAvgRecall(Callback):
    def __init__(self, filepath, monitor1='recall', monitor2='recall_esca', verbose=1, save_best_only=True, mode='max'):
        super(ModelCheckpointAvgRecall, self).__init__()
        self.filepath = filepath
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = -np.Inf if self.mode == 'max' else np.Inf
        self.monitor_op = np.greater if self.mode == 'max' else np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        monitor1_value = logs.get(self.monitor1)
        monitor2_value = logs.get(self.monitor2)
        
        if monitor1_value is not None and monitor2_value is not None:
            avg_recall = (monitor1_value + monitor2_value) / 2
            if self.monitor_op(avg_recall, self.best):
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: {self.monitor1} and {self.monitor2} improved ({self.best:.5f} --> {avg_recall:.5f}). Saving model to {self.filepath}')
                self.best = avg_recall
                self.model.save(self.filepath)

recall_for_class_0 = RecallForClass(class_id=0, name='recall_esca')

class_weights = {0: 1.88, 1: 0.68}

def build_model():
    base_model = EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet')
    base_model.trainable = False
    
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block2a_expand_activation').output)
    
    inputs = Input(shape=(150, 150, 3))
    x = intermediate_layer_model(inputs)
    
    x = KANConv2D(64, 3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.5)(x)
    
    x = KANConv2D(128, 3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

def focal_loss(gamma=2.5, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(fl)
    return focal_loss_fixed

model = build_model()
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(), recall_for_class_0]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
custom_early_stopping = EarlyStoppingByMetric(monitor1='accuracy', monitor2='recall', monitor3='recall_esca', value1=0.98, value2=0.9, value3=0.9)

checkpoint = ModelCheckpointAvgRecall(
    filepath='/content/drive/MyDrive/best_grapeleaf_classifier_kan_enet.keras', 
    monitor1='recall', 
    monitor2='recall_esca', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

history = model.fit(
    train_generator,
    steps_per_epoch=1500,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr, custom_early_stopping, checkpoint]
)

model.save('/content/drive/MyDrive/grapeleaf_classifier_kan_enet.keras')

results = model.evaluate(validation_generator)
loss, accuracy, recall, recall_esca = results[0], results[1], results[2], results[3]
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
print(f'Test recall: {recall}')
print(f'Test recall_esca: {recall_esca}')

predictions = model.predict(validation_generator).ravel()
true_classes = validation_generator.classes
predicted_classes = np.where(predictions > 0.15, 1, 0)

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
