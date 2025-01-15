Traffic Light Classification using CNNs

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify images of traffic lights as either red or green. The model is trained using a dataset of traffic light images, employing data augmentation techniques to improve its robustness and accuracy. The project utilizes TensorFlow and Keras for building and training the CNN model.

Project Features
- Data Augmentation: The model is trained using image data that has been augmented to improve generalization. Augmentation techniques such as rotation, zoom, and horizontal flip are applied to the training images.
- CNN Architecture: The Convolutional Neural Network model includes multiple convolutional layers for feature extraction, max-pooling layers for dimensionality reduction, and fully connected layers for classification. A dropout layer is added for regularization to prevent overfitting.
- Regularization Techniques: Early stopping is implemented to halt training when the validation performance stops improving, and dropout is used in the model to reduce overfitting.
- Image Prediction: The project includes functionality to predict traffic light states (red or green) from external image URLs. The model is used to classify the traffic light in the image and output the corresponding state.
- Model Evaluation: The model is trained for 30 epochs and its performance is evaluated using both training and validation datasets. The final accuracy achieved on the validation set is over 92%.

Technologies Used
- Python: The programming language used for model development and data processing.
- TensorFlow and Keras: Libraries for building, training, and evaluating the CNN model.
- NumPy: Used for numerical operations and manipulating image arrays.
- Matplotlib: For visualizing training performance, including accuracy and loss plots.
- Pillow (PIL): Used for image manipulation and handling external image URLs.
- ImageDataGenerator: From Keras, used to preprocess and augment the images before training.

Results
The model achieves an accuracy of over 92% on the validation dataset. It is capable of accurately classifying traffic lights as either red or green, making it suitable for real-time traffic light detection systems.

Future Improvements
- Extend the model to classify other states of traffic lights, such as yellow.
- Optimize the model for deployment on mobile devices or embedded systems.
- Experiment with different CNN architectures or pre-trained models like VGG or ResNet to improve performance further.
