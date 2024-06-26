import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping
import os
import shutil

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
        # Extract patches from the input
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper()
        )
        
        # Use tf.shape to get dynamic dimensions
        input_shape = tf.shape(inputs)
        patches_shape = tf.shape(patches)
        patch_size = self.kernel_size[0] * self.kernel_size[1] * input_shape[-1]
        
        patches_reshaped = tf.reshape(patches, [-1, patch_size])
        control_points_reshaped = tf.reshape(self.control_points, [-1, self.filters])
        
        # Compute distances
        distances = tf.reduce_sum(tf.square(tf.expand_dims(patches_reshaped, 2) - tf.expand_dims(control_points_reshaped, 0)), axis=1)
        
        # Apply RBF kernel
        gamma = 1.0 / (2.0 * tf.reduce_mean(distances))
        kernel_output = tf.exp(-gamma * distances)
        
        # Reshape kernel output to match conv2d output shape
        kernel_output_reshaped = tf.reshape(kernel_output, [patches_shape[0], patches_shape[1], patches_shape[2], self.filters])
        
        return kernel_output_reshaped

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)
    

# Data Preparation
original_data_dir = '/content/drive/MyDrive/gtprac/original_grape_data'
train_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_train'
test_dir = '/content/drive/MyDrive/gtprac/original_grape_data/binary_test'

# def clear_and_create_dir(directory):
#     if os.path.exists(directory):
#         shutil.rmtree(directory)
#         print(f"Removed existing directory: {directory}")
#     os.makedirs(directory)
#     print(f"Created directory: {directory}")

# clear_and_create_dir(train_dir)
# clear_and_create_dir(test_dir)

# for category in ['healthy', 'esca']:
#     os.makedirs(os.path.join(train_dir, category), exist_ok=True)
#     os.makedirs(os.path.join(test_dir, category), exist_ok=True)
#     print(f"Created subdirectories for category: {category} in both train and test directories")

# def move_files(src_dir, dst_dir, category):
#     print(f"Moving files from {src_dir} to {dst_dir} for category: {category}")
#     for folder in os.listdir(src_dir):
#         folder_path = os.path.join(src_dir, folder)
#         if os.path.isdir(folder_path):
#             for file in os.listdir(folder_path):
#                 file_path = os.path.join(folder_path, file)
#                 if category == 'healthy' and folder != 'ESCA':
#                     shutil.copy(file_path, os.path.join(dst_dir, 'healthy'))
#                     # print(f"Copied {file_path} to {os.path.join(dst_dir, 'healthy')}")
#                 elif category == 'esca' and folder == 'ESCA':
#                     shutil.copy(file_path, os.path.join(dst_dir, 'esca'))
#                     # print(f"Copied {file_path} to {os.path.join(dst_dir, 'esca')}")

# # Combine images to create healthy and esca paths
# move_files(os.path.join(original_data_dir, 'train'), train_dir, 'healthy')
# move_files(os.path.join(original_data_dir, 'train'), train_dir, 'esca')
# move_files(os.path.join(original_data_dir, 'test'), test_dir, 'healthy')
# move_files(os.path.join(original_data_dir, 'test'), test_dir, 'esca')

print("Data preparation complete.")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
print(f"Training data loaded from {train_dir}")

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)
print(f"Validation data loaded from {test_dir}")

def build_model():
    inputs = Input(shape=(150, 150, 3))
    x = KANConv2D(64, 3, padding='same')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = KANConv2D(128, 3, padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    x = KANConv2D(256, 3, padding='same')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

# Adjust class weights to reduce ESCA false negatives
class_weights = {0: 1.0, 1: 2.3}

# Compile the model with class weights and learning rate adjustment
model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with class weights and early stopping
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Save keras model
model.save('/content/drive/MyDrive/grapeleaf_classifier_kan_best.keras')

# Evaluate
loss, accuracy = model.evaluate(validation_generator)
print(f'Test accuracy: {accuracy}, Test loss: {loss}')
