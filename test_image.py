from hack import WasteClassifier  # Import from your existing code

def test_new_image(image_path):
    """
    Test a single new image using our trained model
    """
    # Load the trained model
    classifier = WasteClassifier('waste_classifier_model.h5')
    
    # Make prediction
    class_name, confidence = classifier.predict(image_path)
    
    # Print results
    print(f"\nResults for image: {image_path}")
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
    # Ask user for image path
    image_path = input("Enter the path to your image (e.g., /Users/user/Downloads/bottle.jpg): ")
    test_new_image(image_path)
