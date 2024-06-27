import matplotlib.pyplot as plt

# Data from the training logs
epochs = list(range(1, 16))
training_loss = [0.8480, 0.7657, 0.7170, 0.6574, 0.5970, 0.5876, 0.5476, 0.5476, 0.5296, 0.5297, 0.5171, 0.5177, 0.5099, 0.5034, 0.4974]
training_accuracy = [0.7293, 0.7349, 0.7420, 0.7460, 0.7618, 0.7607, 0.7766, 0.7782, 0.7926, 0.7945, 0.7998, 0.8021, 0.7985, 0.8134, 0.8098]
validation_loss = [0.5629, 0.4964, 0.6062, 0.4143, 0.3464, 0.3556, 0.3699, 0.3422, 0.3348, 0.3321, 0.3317, 0.3085, 0.3390, 0.3193, 0.3276]
validation_accuracy = [0.7341, 0.7346, 0.7341, 0.7496, 0.8404, 0.8211, 0.8443, 0.8465, 0.8604, 0.8548, 0.8460, 0.8659, 0.8609, 0.8620, 0.8476]

# Create a figure with two subplots (stacked vertically)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot training loss and accuracy
ax1.plot(epochs, training_loss, label='Training Loss', color='blue')
ax1.plot(epochs, training_accuracy, label='Training Accuracy', color='green')
ax1.set_ylabel('Value')
ax1.legend(loc='lower right')
ax1.set_title('Training Loss and Accuracy Over Epochs')
ax1.set_ylim(0, 1)  # Set y-axis limits from 0 to 1

# Plot validation loss and accuracy
ax2.plot(epochs, validation_loss, label='Validation Loss', color='red')
ax2.plot(epochs, validation_accuracy, label='Validation Accuracy', color='orange')
ax2.set_ylabel('Value')
ax2.set_xlabel('Epoch')
ax2.legend(loc='lower right')
ax2.set_title('Validation Loss and Accuracy Over Epochs')
ax2.set_ylim(0, 1)  # Set y-axis limits from 0 to 1

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
