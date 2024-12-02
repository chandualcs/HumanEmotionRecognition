# HumanEmotionRecognition
**Model Overview:**
The model is a Convolutional Neural Network (CNN) based on the ResNet architecture, designed for a multi-class classification task using the AffectNet dataset. The model is intended to classify images into one of seven classes based on facial expressions.
# Model Architecture:
The model architecture is based on the ResNet design:
1.	Input Layer: Accepts images with a shape of (64, 64, 3).
2.	Convolutional Layers: Employs convolutional layers with kernel sizes of 7x7 and 3x3 with batch normalization and ReLU activation.
3.	Pooling Layers: Utilizes max-pooling layers to downsample the spatial dimensions.
4.	Residual Blocks: Utilizes residual blocks to capture features.
5.	Global Average Pooling: Averages the spatial dimensions to a single vector.
6.	Output Layer: A dense layer with a softmax activation for multi-class classification.
# Model Training:
•	Optimization Algorithm: Adam optimizer was used with the categorical cross-entropy loss function.
•	Training Data: Train and validation images were loaded and pre-processed from the AffectNet dataset.
•	Training Duration: The model was trained for 50 epochs with a batch size of 32.
# Model Performance:
•	Validation Accuracy: The model achieved a validation accuracy of 0.8862
•	Loss: The loss value after training was 0.4906
•	Mean Squared Error (MSE): 0.11110322177410126
# Results of Experimentations of the model with different combinations of hyperparameters:

	Learning rate 0.001 and number of epochs = 25:
	loss: 0.9517 - accuracy: 0.6883 - val_loss: 1.8166 - val_accuracy: 0.4089
	(R2): 0.12184411619043019
	Learning rate 0.001 and number of epochs = 30:
	loss: 0.8565 - accuracy: 0.7305 - val_loss: 1.8782 - val_accuracy: 0.4156
	“Such many experimentations are done and such experimental results 
are shown below”

# Evaluation:
The model's performance was evaluated based on the validation set, achieving an accuracy of  0.4610 and a loss of 2.1360
![Loos_graphs](https://github.com/chandualcs/HumanEmotionRecognition/blob/main/images/loss_graphs.png)

