import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

mixed_precision.set_global_policy('mixed_float16')

class KANConv2D(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', **kwargs):
        super(KANConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.bias = self.add_weight(shape=(self.filters,),
                                    initializer='zeros',
                                    name='bias')
        self.control_points = self.add_weight(shape=self.kernel_shape,
                                              initializer='glorot_uniform',
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

original_data_dir = '/content/drive/MyDrive/gtprac/original_grape_data'
train_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_train'
test_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=6,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=6,
    class_mode='binary'
)

class_weights = {0: 3.4, 1: 1.0}

base_model = EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet')
base_model.trainable = False

def build_model():
    base_model = EfficientNetB0(include_top=False, input_shape=(150, 150, 3), weights='imagenet')
    base_model.trainable = False
    
    # an intermediate layer of EfficientNetB0
    intermediate_layer_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('block2a_expand_activation').output)
    
    inputs = Input(shape=(150, 150, 3))
    x = intermediate_layer_model(inputs, training=False)
    
    x = KANConv2D(64, 3, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  
    print(f'Shape after first block: {x.shape}')

    x = KANConv2D(128, 3, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) 
    print(f'Shape after second block: {x.shape}')
    
    x = KANConv2D(256, 3, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)  
    print(f'Shape after third block: {x.shape}')
    
    x = KANConv2D(512, 3, padding='same')(x)
    print(f'Shape after fourth block: {x.shape}')
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

model = build_model()
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

model.save('/content/drive/MyDrive/grapeleaf_classifier_kan_enet.keras')

loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')

predictions = model.predict(validation_generator).ravel()
true_classes = validation_generator.classes
predicted_classes = np.where(predictions > 0.5, 1, 0)
print(classification_report(true_classes, predicted_classes))

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, thresholds = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
roc_curve_path = os.path.join(original_data_dir, 'roc_curve_enet.png')
plt.savefig(roc_curve_path)
print(f'ROC curve saved to {roc_curve_path}')
