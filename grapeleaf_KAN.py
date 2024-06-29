import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Layer, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# mixed_precision.set_global_policy('mixed_float16')

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
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)
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

print("Data preparation complete.")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
print(f"Training data loaded from {train_dir}")

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
print(f"Validation data loaded from {test_dir}")

print("Class distribution in training data:")
print(train_generator.class_indices)
print(np.bincount(train_generator.classes))

print("Class distribution in validation data:")
print(validation_generator.class_indices)
print(np.bincount(validation_generator.classes))

def build_model():
    inputs = Input(shape=(150, 150, 3)) 
    x = KANConv2D(64, 3, padding='same', kernel_regularizer=l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = KANConv2D(128, 3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = KANConv2D(256, 3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = KANConv2D(512, 3, padding='same', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

class_weights = {0: 1.88, 1: 0.68}  

# experiement with focal loss -> be sure to change in compile
def focal_loss(gamma=2.5, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * tf.keras.backend.pow((tf.keras.backend.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(fl)
    return focal_loss_fixed

model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipvalue=0.5)  
model.compile(
    optimizer=optimizer,
    loss=focal_loss(),
    # loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
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

model.save('/content/drive/MyDrive/grapeleaf_classifier_kan_best.keras')

print("Starting model evaluation...")
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')

print("Starting predictions...")
validation_generator.reset() 
predictions = model.predict(validation_generator)

print(f'Predictions range: {np.min(predictions)} to {np.max(predictions)}')

threshold = 0.50  
predicted_classes = np.where(predictions > threshold, 1, 0)

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

print("Predictions for each image:")
for i in range(len(predictions)):
    image_path = validation_generator.filepaths[i]
    image_name = os.path.basename(image_path)
    
    print(f"Image: {image_name}, Prediction: {predictions[i][0]:.4f}, Predicted class: {'Healthy' if predicted_classes[i] == 1 else 'Esca'}, True class: {'Healthy' if true_classes[i] == 1 else 'Esca'}")

print("Classification report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
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

roc_curve_path = os.path.join(original_data_dir, 'roc_curve.png')
plt.savefig(roc_curve_path)
print(f'ROC curve saved to {roc_curve_path}')

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f'Optimal threshold: {optimal_threshold}')
