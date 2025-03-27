import os
import shutil
import random
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def extract_dataset():
    zip_path = '(Path to the trashnet file)trashnet-master/data/dataset-resized.zip'
    extract_path = '(path to the trashnet file)/trashnet-master/data'
    
    print("Extracting dataset-resized.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed!")

def organize_dataset():
    print("Organizing dataset into training and validation sets...")
    
    # Extracting the dataset if needed
    if not os.path.exists('(path to the trashnet file)/trashnet-master/data/dataset-resized'):
        print("Categories not found, extracting dataset first...")
        extract_dataset()
    
    # Use the path to the dataset-resized folder
    source_dir = '(path to the trashnet files)/trashnet-master/data/dataset-resized'
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} directory not found!")
        return
    
    print(f"Found dataset at: {source_dir}")
    print(f"Contents of {source_dir}:")
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            print(f"  {item}/: {len(os.listdir(item_path))} files")
    
    # 1. Create the main directories
    train_dir = 'training_data'
    val_dir = 'validation_data'
    
    # Clean up existing directories
    for dir_path in [train_dir, val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    # 2. Create category subdirectories
    for dir_path in [train_dir, val_dir]:
        for category in ['recyclable', 'trash']:
            os.makedirs(os.path.join(dir_path, category), exist_ok=True)
    
    # 3. Define which TrashNet categories go into our categories
    recyclable_categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal']
    trash_categories = ['trash']
    
    # 4. Process each category
    for category in os.listdir(source_dir):
        if category in recyclable_categories:
            target_category = 'recyclable'
        elif category in trash_categories:
            target_category = 'trash'
        else:
            continue
        
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print(f"Warning: No image files found in {category_path}")
            continue
            
        random.shuffle(files)
        
        split_point = int(len(files) * 0.2)
        validation_files = files[:split_point]
        training_files = files[split_point:]
        
        print(f"Processing {category} -> {target_category}...")
        print(f"  Training: {len(training_files)} files")
        print(f"  Validation: {len(validation_files)} files")
        
        for file in validation_files:
            shutil.copy2(
                os.path.join(category_path, file),
                os.path.join(val_dir, target_category, file)
            )
        
        for file in training_files:
            shutil.copy2(
                os.path.join(category_path, file),
                os.path.join(train_dir, target_category, file)
            )
    
    return train_dir, val_dir

class WasteClassifier:
    def __init__(self, model_path=None):
        self.model = self.build_model()
        if model_path and os.path.exists(model_path):
            self.model.load_weights(model_path)
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',  # since we have 2 categories
            metrics=['accuracy']
        )
    
    def build_model(self):
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification (recyclable vs trash)
        ])
        return model
    
    def train(self, train_dir, val_dir, epochs=10, batch_size=32):
        # Data Augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',  # for binary classification
            shuffle=True
        )

        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',  # for binary classification
            shuffle=False
        )

        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )
        
        return history
    
    def predict(self, image_path):
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)
        img_array /= 255.0
        
        # Make prediction
        prediction = self.model.predict(img_array)
        class_name = 'recyclable' if prediction[0] > 0.5 else 'trash'
        confidence = prediction[0] if prediction[0] > 0.5 else 1 - prediction[0]
        
        return class_name, float(confidence)
    
    def save_model(self, model_path):
        self.model.save_weights(model_path)

def main():
    # Step 1: Organize dataset
    print("Step 1: Organizing dataset...")
    train_dir, val_dir = organize_dataset()
    
    # Step 2: Create and train the model
    print("\nStep 2: Training model...")
    classifier = WasteClassifier()
    history = classifier.train(train_dir, val_dir, epochs=10)
    
    # Step 3: Save the model
    print("\nStep 3: Saving model...")
    classifier.save_model('waste_classifier_model.h5')
    
    # Step 4: Test the model
    print("\nStep 4: Testing model...")
    # Test on a few images from the validation set
    val_recyclable_dir = os.path.join(val_dir, 'recyclable')
    val_trash_dir = os.path.join(val_dir, 'trash')
    
    # Test one image from each category
    if os.path.exists(val_recyclable_dir) and os.listdir(val_recyclable_dir):
        test_recyclable = os.path.join(val_recyclable_dir, os.listdir(val_recyclable_dir)[0])
        class_name, confidence = classifier.predict(test_recyclable)
        print(f"\nTest recyclable image: {test_recyclable}")
        print(f"Predicted: {class_name} with {confidence:.2%} confidence")
    
    if os.path.exists(val_trash_dir) and os.listdir(val_trash_dir):
        test_trash = os.path.join(val_trash_dir, os.listdir(val_trash_dir)[0])
        class_name, confidence = classifier.predict(test_trash)
        print(f"\nTest trash image: {test_trash}")
        print(f"Predicted: {class_name} with {confidence:.2%} confidence")


if __name__ == "__main__":
    main()
