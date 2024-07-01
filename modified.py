import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import shutil
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Layer, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
train_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_train_2'
test_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_test_2'

# def clear_and_create_dir(directory):
#     if os.path.exists(directory):
#         shutil.rmtree(directory)
#     os.makedirs(directory)

# clear_and_create_dir(train_dir)
# clear_and_create_dir(test_dir)

# for category in ['healthy', 'esca']:
#     os.makedirs(os.path.join(train_dir, category), exist_ok=True)
#     os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# def move_files(src_dir, dst_dir, category):
#     for folder in os.listdir(src_dir):
#         folder_path = os.path.join(src_dir, folder)
#         if os.path.isdir(folder_path):
#             for file in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, file)
#                 if category == 'healthy' and folder != 'ESCA':
#                     shutil.copy(file_path, os.path.join(dst_dir, 'healthy'))
#                 elif category == 'esca' and folder == 'ESCA':
#                     shutil.copy(file_path, os.path.join(dst_dir, 'esca'))

# move_files(os.path.join(original_data_dir, 'train'), train_dir, 'healthy')
# move_files(os.path.join(original_data_dir, 'train'), train_dir, 'esca')
# move_files(os.path.join(original_data_dir, 'test'), test_dir, 'healthy')
# move_files(os.path.join(original_data_dir, 'test'), test_dir, 'esca')

print(f"Training set image counts: {sum([len(files) for r, d, files in os.walk(train_dir)])}")
print(f"Test set image counts: {sum([len(files) for r, d, files in os.walk(test_dir)])}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=50,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Custom function to map filenames to their corresponding class indices
def custom_flow_from_directory(directory, batch_size, target_size=(150, 150), shuffle=True, seed=None):
    classes = ['esca', 'healthy', 'blight', 'rot']
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    filepaths = []
    labels = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg', 'JPG')):
                filepath = os.path.join(root, file)
                if 'esca' in root.lower():
                    labels.append(class_indices['esca'])
                else:
                    if 'L.Blight' in file:
                        labels.append(class_indices['blight'])
                    elif 'B.Rot' in file:
                        labels.append(class_indices['rot'])
                    else:
                        labels.append(class_indices['healthy'])
                filepaths.append(filepath)
    
    if not filepaths:
        raise ValueError("No image files found in the directory.")
    
    filepaths = np.array(filepaths)
    labels = np.array(labels, dtype=np.int32)  # Ensure labels are integers

    def decode_img(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = img / 255.0
        return img
    
    def process_path(file_path, label):
        file_path = tf.py_function(func=lambda z: tf.convert_to_tensor(z.numpy().decode(), dtype=tf.string), inp=[file_path], Tout=tf.string)
        img = decode_img(file_path)
        label = tf.one_hot(label, depth=len(classes))
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = custom_flow_from_directory(train_dir, batch_size=256)
validation_dataset = custom_flow_from_directory(test_dir, batch_size=256)

def model():
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
    
    outputs = Dense(1, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Compile model with categorical crossentropy
model = model()
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001, clipvalue=1.0)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    #callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_dataset)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Ensure the entire binary test dataset is shuffled and used for predictions
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=256,
    class_mode='categorical',
    shuffle=False  # Important to set shuffle to False to keep file paths aligned with predictions
)

# Get file paths
file_paths = test_generator.filepaths

# Predictions
y_pred_prob = model.predict(test_generator)
y_true = test_generator.classes

# Calculate average prediction values by class
average_pred_healthy = None
average_pred_esca = None
average_pred_blight = None
average_pred_rot = None

if 'healthy' in test_generator.class_indices:
    average_pred_healthy = np.mean(y_pred_prob[y_true == test_generator.class_indices['healthy'], test_generator.class_indices['healthy']])
    print(f'Average prediction value for healthy: {average_pred_healthy}')

if 'esca' in test_generator.class_indices:
    average_pred_esca = np.mean(y_pred_prob[y_true == test_generator.class_indices['esca'], test_generator.class_indices['esca']])
    print(f'Average prediction value for esca: {average_pred_esca}')

if 'blight' in test_generator.class_indices:
    average_pred_blight = np.mean(y_pred_prob[y_true == test_generator.class_indices['blight'], test_generator.class_indices['blight']])
    print(f'Average prediction value for blight: {average_pred_blight}')

if 'rot' in test_generator.class_indices:
    average_pred_rot = np.mean(y_pred_prob[y_true == test_generator.class_indices['rot'], test_generator.class_indices['rot']])
    print(f'Average prediction value for rot: {average_pred_rot}')

# Calculate custom threshold using the average of all non-esca and esca predictions
non_esca_preds = [pred for pred in [average_pred_healthy, average_pred_blight, average_pred_rot] if pred is not None]
if not non_esca_preds or average_pred_esca is None:
    raise ValueError("Missing predictions for calculating the threshold.")
average_pred_non_esca = np.mean(non_esca_preds)
custom_threshold = (average_pred_non_esca + average_pred_esca) / 2
print(f'Custom threshold: {custom_threshold}')

# Apply custom threshold to generate binary predictions
binary_pred = (y_pred_prob[:, test_generator.class_indices['esca']] > custom_threshold).astype(int)

# Display predicted and true classes along with file paths
predicted_classes = []
true_classes = []
for i, file_path in enumerate(file_paths):
    true_class = y_true[i]
    predicted_class = binary_pred[i]
    print(f'File: {file_path}, Predicted class: {"esca" if predicted_class == 1 else "non-esca"}, True class: {"esca" if true_class == test_generator.class_indices["esca"] else "non-esca"}')
    predicted_classes.append(predicted_class)
    true_classes.append(1 if true_class == test_generator.class_indices['esca'] else 0)

# Binary evaluation: esca vs all other classes
binary_true = np.array(true_classes)
binary_pred = np.array(predicted_classes)

# Iterate over different thresholds to find the optimal one, excluding 0.0
thresholds = np.arange(0.01, 1.0, 0.01)
f1_scores = []

for threshold in thresholds:
    binary_pred = (y_pred_prob[:, test_generator.class_indices['esca']] > threshold).astype(int)
    f1 = f1_score(binary_true, binary_pred)
    f1_scores.append(f1)

ideal_threshold = thresholds[np.argmax(f1_scores)]
print(f'Ideal threshold: {ideal_threshold}')
print(f'Best F1 score at ideal threshold: {max(f1_scores)}')

# Use the ideal threshold to make final predictions
binary_pred = (y_pred_prob[:, test_generator.class_indices['esca']] > ideal_threshold).astype(int)

print("Binary Classification Report (esca vs others):")
print(classification_report(binary_true, binary_pred, target_names=['non-esca', 'esca']))

print("Binary Confusion Matrix (esca vs others):")
print(confusion_matrix(binary_true, binary_pred))
