# Hand Gesture Recognition using Convolutional Neural Networks (CNN)

This project focuses on recognizing hand gestures using a Convolutional Neural Network (CNN) model. The dataset used for this project is the **LeapGestRecog** dataset, which contains images of various hand gestures. The goal of this project is to classify different hand gestures with high accuracy, providing a step forward in gesture-based human-computer interaction.

## Project Overview

Hand gesture recognition is a crucial component in building interfaces that allow intuitive control systems using gestures. This project explores the application of deep learning to recognize and classify hand gestures into different categories. The model achieves over 99% accuracy on the test set, demonstrating its efficiency and potential for real-world applications.

## Dataset

The **LeapGestRecog** dataset consists of images representing 10 distinct hand gestures, captured in grayscale. Each image belongs to one of the following gesture categories:

- Palm
- Fist
- Thumbs Up
- Ok
- L-shape
- Palm Moved
- C-shape
- Index Pointing
- Fist Moved
- Down Gesture

Each class contains images captured under varying conditions, providing diverse input data to train the CNN model. All images are resized to 128x128 pixels for consistent input into the neural network.

## Model Architecture

The CNN model is built using **Keras** and **TensorFlow**. The architecture consists of multiple convolutional layers followed by max-pooling and fully connected (dense) layers. Below is an overview of the architecture:

- **Input Layer**: Grayscale images of size 128x128.
- **Convolutional Layers**: Three convolutional layers with 32, 64, and 128 filters, respectively, followed by ReLU activation and max-pooling.
- **Dropout Layers**: Used to prevent overfitting by randomly dropping units during training.
- **Fully Connected Layers**: Two dense layers, where the first one has 128 units and the output layer has 10 units (one for each class).
- **Output Layer**: Uses the Softmax activation function for multiclass classification.

## Training and Evaluation

- **Training**: The model is trained on 80% of the dataset using the Adam optimizer, with a categorical cross-entropy loss function. The training is performed over 25 epochs.
- **Validation**: 10% of the data is used for validation during training to monitor the model’s performance and prevent overfitting.
- **Testing**: The remaining 10% of the data is used to evaluate the final performance of the model.

### Performance Metrics

- **Training Accuracy**: 99.8%
- **Validation Accuracy**: 99.5%
- **Test Accuracy**: 99.15%
- **Loss**: The model achieves a low loss on both training and test sets, indicating high generalization capacity.

## Installation and Setup

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/sihemaf/PRODIGY_ML_04.git
   
2. Change into the project directory:
   ```bash
   cd PRODIGY_ML_04
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
## Usage
1. Ensure the dataset is organized in the correct structure as described in the project.

2. To train the model, use the following command: 
   ```bash
   python train_model.py
   
3. Once the model is trained, you can evaluate it on the test set by running:
   ```bash
   python evaluate_model.py
   
4. You can also visualize the training process and the accuracy/loss curves:
   ```bash
    python plot_training_curves.py
   
## Results and Visualizations
The following plots demonstrate the training and validation accuracy and loss over the epochs:

- **Accuracy Curve**: The model achieves near-perfect accuracy with a smooth convergence.
- **Loss Curve**: The loss function converges consistently, showing minimal signs of overfitting.
  Additionally, confusion matrices and classification reports can be generated to better understand the model's performance on each gesture class.

## Future Work
- Implement real-time gesture recognition using a webcam.
- Experiment with different CNN architectures or transfer learning to improve accuracy further.
- Explore data augmentation techniques to make the model more robust to different lighting conditions and hand positions.
- Test the model in a real-world application, such as a gesture-based control system.
## Acknowledgments
- **LeapGestRecog** dataset contributors for providing a diverse collection of hand gesture images.
- The open-source community for tools and resources that supported this project.
- Keras and TensorFlow for making deep learning accessible and efficient.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

