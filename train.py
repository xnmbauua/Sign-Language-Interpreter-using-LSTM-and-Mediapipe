import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Load sequence data
X = np.load('X_seq.npy')  # shape: (samples, 30, 126)
y = np.load('y_seq.npy')  # shape: (samples,)

# One-hot encode labels
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)
num_classes = y_encoded.shape[1]

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Build LSTM model with BatchNormalization
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(30, 126)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save best model during training
checkpoint = ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint]
)

# Save final model
model.save('ultimate_model.keras')
print("✅ Model training complete. Saved final model to 'ultimate_model.keras'.")
