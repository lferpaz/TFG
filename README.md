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
### Folder Structure - src 📁
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
### Folder Structure - docs 📁
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
### Folder Structure - results 📁
 - `results/`: includes a collection of outcomes achieved during the project, such as:
    - Performance evaluations of models using different metrics.
    - Results obtained from various image preprocessing techniques.
    - Outputs generated by manipulation area detection methods.
    - Details about the architecture of the neural network used.
### Folder Structure - test 📁

- `tests/`: contains the following content:
  - This folder includes images used for testing purposes with the final model.
  - The images in this folder consist of a combination of viral images sourced from the internet and personal photos.
  - These images were utilized to evaluate the model's performance and assess its generalization capability.

## 	:toolbox: Getting Started

To run the system, you need to install Python 3 and TensorFlow. You can use the following commands to install the required packages:
pip install tensorflow

Once you have installed the required packages, you can run the system by running the following command:
python src/main.py


For more detailed instructions on how to install and run the system, please refer to the user manual in the `docs/` directory.

## Brief Explanation of Model Operation 😃

The following provides a brief explanation of the process for classifying an initial image using the model and determining if it has been modified, while also identifying the area of modification.

1. **Image Preprocessing**
   - Our input is an RGB image that requires preprocessing before passing it to the model for analysis.
   - The first step is resizing the image to a specific size, typically the size the model was trained on, such as a 128x128 pixel image.
   - Next, we apply the Error Level Analysis (ELA) algorithm to enhance the image's features.

2. **Image Transformation**
   - To improve the model's performance, we convert the processed image into a matrix before feeding it to the model for classification.
   - This matrix is then used as input for the classification model.

3. **Model Classification and Output**
   - The model performs the classification on the image and returns a vector with two values.
   - These values represent the prediction percentages for two existing classes: authentic or manipulated.
   - The model's output provides an indication of the likelihood that the image belongs to either class.

4. **Result Visualization**
   - For visualization purposes, a function is applied that utilizes one of the layers generated by the model during feature extraction.
   - A heat map is applied to the image, highlighting the identified area of modification.
   - Finally, the resulting image, emphasizing the identified area of modification, is displayed.

This process enables the classification of an initial image, determination of potential modifications, and identification of the modified area using preprocessing techniques, model classification, and result visualization.

<img src="https://github.com/lferpaz/TFG/assets/57639021/d028dbb3-a87b-4cd6-abdf-db7eb21fc599" alt="image" width="600" height="400">





## :handshake: Contact

If you have any questions or feedback about the project, please feel free to contact me at [your-email-address]. I am happy to discuss any aspect of the project with you.

## :gem: Acknowledgements

I would like to thank my supervisor [supervisor-name] for his/her guidance and support throughout the project. I would also like to thank [colleagues-names] for their valuable feedback and suggestions.

## :warning: License

This project is licensed under the [license-name] license. For more information, please see the LICENSE file in the root directory of the repository.

