# Detection of modified images or videos using Neural Networks

This repository contains the development of the final degree project "Detection of modified images or videos using Neural Networks" for the Computer Engineering degree at the Autonomous University of Barcelona.

## :abacus: Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## :eye: Overview

The main objective of this project is the development of a system for detecting modified images or videos using neural networks. This system can be used to identify tampered media, which is an important task in fields such as digital forensics, journalism, and social media analysis.

The system is developed using Python and TensorFlow, and it is based on state-of-the-art deep learning architectures such as convolutional neural networks and recurrent neural networks.

## :key: Repository structure

The repository is organized as follows:
### Folder Structure - src
`src/`: contains the source code of the system, including the neural network models and the pre-processing and post-processing modules.
The following is an explanation of the project's structure and how it has been organized in the src folder of my GitHub repository:

1. **01-model.ipynb**
   - Contains the structure of the neural network used to generate the model.
   - Includes functions for building and training the model.
   - Loads the dataset images (both real and manipulated) and sets the training parameters (number of epochs, image size, batch size).
   - Saves the model in the corresponding folder.

2. **02-Techniques-of-detection-of-manipulations-in-images.ipynb**
   - Contains different techniques or algorithms applied to images to detect modified areas.
   - Includes techniques such as heat map based on pixel distribution, oriented gradient histogram, local gradient, shadow variation, color analysis, and lighting inconsistency.
   - These techniques support the feature extraction function performed by the model and provide a contrast with the model's predictions.
   - They add value to the project by allowing the identification of modified areas in an image using existing algorithms.

3. **03-Cross-validation.ipynb**
   - Contains the code for performing cross-validation and evaluating the model's performance.

4. **04-Hyperparameter-search.ipynb**
   - Contains the hyperparameter search to optimize the model's performance.
   - Uses the Optuna library for an iterative search of the best parameters that result in improved accuracy.

5. **preprocess.py**
   - Contains functions for preprocessing images before training the model.
   - Includes other functions used for various tests and obtaining different results for the available neural network architecture.

6. **utils.py**
   - Contains useful functions for data handling.
### Folder Structure - docs
- `docs/`: contains the documentation of the project, including the report, slides, and user manual
1. **Bibliography Folder**
   - This folder contains various papers from renowned experts in the field of image manipulation recognition.
   - These papers were instrumental in the project's development as they provided diverse perspectives that formed the basis for building the model.

2. **Dossier Folder**
   - Inside this folder, you will find different project deliverables and the final project report.
   - It includes the documentation related to the project's milestones and the comprehensive summary of the project in the form of a report.

3. **Hyperparameters Folder**
   - This folder stores the results of the various hyperparameter searches and experiments conducted.
   - Due to the algorithmic complexity, multiple datasets were utilized, varying the quantity of data used for experimentation.
   - The folder contains estimations and findings that helped identify parameter values for optimizing the model's performance.
- `tests/`: contains the testing scripts and results for the system.

## 	:toolbox: Getting Started

To run the system, you need to install Python 3 and TensorFlow. You can use the following commands to install the required packages:
pip install tensorflow

Once you have installed the required packages, you can run the system by running the following command:
python src/main.py


For more detailed instructions on how to install and run the system, please refer to the user manual in the `docs/` directory.

## :handshake: Contact

If you have any questions or feedback about the project, please feel free to contact me at [your-email-address]. I am happy to discuss any aspect of the project with you.

## :gem: Acknowledgements

I would like to thank my supervisor [supervisor-name] for his/her guidance and support throughout the project. I would also like to thank [colleagues-names] for their valuable feedback and suggestions.

## :warning: License

This project is licensed under the [license-name] license. For more information, please see the LICENSE file in the root directory of the repository.

