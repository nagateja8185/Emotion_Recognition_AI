import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

# --- AUTO-DETECT PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "images", "fer2013", "train")
TEST_DIR  = os.path.join(BASE_DIR, "data", "images", "fer2013", "test")
OUT_MODEL = os.path.join(BASE_DIR, "models", "image_emotion.h5")

print("📂 Training folder:", TRAIN_DIR)
print("📂 Testing folder :", TEST_DIR)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

IMG_SIZE = 48
BATCH_SIZE = 32

def lr_schedule(epoch, lr):
    """Learning rate scheduler - decay by 0.5 every 10 epochs"""
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    else:
        return lr * 0.25

def build_model(input_shape=(48, 48, 1), num_classes=7):
    """Enhanced CNN architecture with better regularization and deeper layers"""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape, 
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                     kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    # Enhanced data augmentation for better generalization
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.1
    )

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=BATCH_SIZE,
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=BATCH_SIZE
    )

    num_classes = len(train_gen.class_indices)
    print("Detected emotion classes:", train_gen.class_indices)

    model = build_model((IMG_SIZE, IMG_SIZE, 1), num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    callbacks = [
        LearningRateScheduler(lr_schedule),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max'),
        tf.keras.callbacks.ModelCheckpoint(OUT_MODEL, monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    print("🚀 Starting training with enhanced architecture...")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=50, 
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Training finished.")

    # Evaluate on test set
    results = model.evaluate(test_gen)
    print(f"\n🧪 Test Results:")
    print(f"   Loss: {results[0]:.3f}")
    print(f"   Accuracy: {results[1]:.3f} ({results[1]*100:.2f}%)")
    if len(results) > 2:
        print(f"   Precision: {results[2]:.3f}")
        print(f"   Recall: {results[3]:.3f}")

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    model.save(OUT_MODEL)
    print("✅ Model saved to", OUT_MODEL)
    
    # Save class indices for reference
    import json
    class_indices_path = os.path.join(BASE_DIR, "models", "image_class_indices.json")
    with open(class_indices_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"✅ Class indices saved to {class_indices_path}")

if __name__ == "__main__":
    main()
