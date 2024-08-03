# Deadlift-ESAP

## Overview
Deadlift-ESAP is a project aimed at creating and training a neural network to determine proper deadlift form. The project includes scripts to generate linkage diagrams from deadlift videos, process these diagrams into data models, and use these models to train a neural network capable of distinguishing good form from bad form in deadlifting.

## Features
- **Linkage Diagram Generation**: Convert deadlift videos into biomechanical linkage diagrams.
- **Pre-processing**: Process raw video data into linkage diagrams for efficient storage and accurate training.
- **Neural Network Training**: Train a neural network to evaluate deadlift form using the generated linkage data.
- **Evaluation**: Assess new deadlift videos to determine form accuracy.

## Requirements
- OpenCV
- OpenPose
- TensorFlow
- Keras
- OpenAI

## File Descriptions

### `network.py`
This file contains the implementation of the neural network used for evaluating deadlift form. It includes the architecture of the model, the training process, and the evaluation metrics.

#### Code Snippet
(Insert code snippet here)

### `pose.py`
This file includes the script to generate linkage diagrams from deadlift videos using OpenPose. It processes raw video data to identify joint points and create a full linkage representation of the deadlift motion.

#### Example GIF
(Insert example GIF here)

### `testmodel.py`
This script is used to test the trained neural network on new deadlift videos. It loads the trained model, processes new video data into linkage diagrams, and evaluates the deadlift form using the neural network.

#### Evaluation Results
(Insert evaluation results here)

## Future Work
- **App Development**: Bundle the project into a user-friendly app for broader accessibility.
- **Expansion to Other Exercises**: Extend the model to analyze other compound lifts and exercises.

## Acknowledgements
The development of this project utilized the following libraries and tools:
- OpenCV
- OpenPose
- TensorFlow
- Keras
- OpenAI

Thank you for using Deadlift-ESAP! For any issues or contributions, please feel free to raise an issue or submit a pull request on our GitHub repository.
