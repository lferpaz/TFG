# Detection of modified images or videos using Neural Networks
![image](https://github.com/lferpaz/TFG/assets/57639021/6b6f75ce-514c-4746-914a-74da5af750ec)

This repository contains the development of the final degree project titled "Detection of Modified Images or Videos using Neural Networks" for the Computer Engineering degree at the Autonomous University of Barcelona.

The project focuses on the application of neural networks for detecting and identifying modified images or videos. It explores the use of convolutional neural networks (CNNs) to analyze and classify visual data based on the presence of modifications. The goal is to develop a robust and accurate model that can differentiate between authentic and manipulated images or videos.

Feel free to explore the repository to learn more about the project and its development. If you have any questions or feedback, please don't hesitate to reach out.

Thank you for your interest in this project!

## Table of contents :abacus:

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Repository structure](#neural-network-architecture)
- [Brief Explanation of Model Operation](#brief-explanation-of-model-operation)
- [Model metrics of interest](#model-metrics-of-interest)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [License](#license)

##  Overview 👁️

The main objective of this project is the development of a system for detecting modified images or videos using neural networks. This system can be used to identify tampered media, which is an important task in fields such as digital forensics, journalism, and social media analysis.

The system is developed using Python and TensorFlow, and it is based on state-of-the-art deep learning architectures such as convolutional neural networks and recurrent neural networks.

## Repository structure 🔑

The repository is organized as follows:
#### Folder Structure - src 📁
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
#### Folder Structure - docs 📁
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
#### Folder Structure - results 📁
 - `results/`: includes a collection of outcomes achieved during the project, such as:
    - Performance evaluations of models using different metrics.
    - Results obtained from various image preprocessing techniques.
    - Outputs generated by manipulation area detection methods.
    - Details about the architecture of the neural network used.
#### Folder Structure - test 📁

- `tests/`: contains the following content:
  - This folder includes images used for testing purposes with the final model.
  - The images in this folder consist of a combination of viral images sourced from the internet and personal photos.
  - These images were utilized to evaluate the model's performance and assess its generalization capability.

## Getting Started 🧰

To run the system, you need to install Python 3 and TensorFlow. You can use the following commands to install the required packages:
pip install tensorflow

Once you have installed the required packages, you can run the system by running the following command:
python src/main.py


For more detailed instructions on how to install and run the system, please refer to the user manual in the `docs/` directory.
## Neural Network Architecture 🏗️
The neural network architecture consists of the following layers:

- Input layer that receives RGB images of size 128x128 pixels.
- Two convolutional layers with 32 filters each, using the ReLU activation function.
- MaxPooling layer to reduce spatial dimension.
- Dropout layer to prevent overfitting.
- Two additional sets of convolutional and MaxPooling layers with 128 filters each.
- Flatten layer to convert the output into a one-dimensional vector.
- Fully connected layer with 512 neurons and ReLU activation function.
- Dropout layer to further prevent overfitting.
- Output layer with 2 neurons and softmax activation function for classifying authentic or manipulated images.

The structure of the neural network is designed to effectively process and extract features from images. It follows the following principles:

- Convolutional layers detect local features.
- MaxPooling layers reduce spatial dimensions.
- Dense layers capture high-level relationships in the data.
- Dropout regularization improves model generalization.
- Softmax activation in the output layer allows for classification.

This architecture enables the model to effectively classify and identify whether an input image has been modified, as well as identify the area of modification.

<img src="https://github.com/lferpaz/TFG/assets/57639021/35dc3cfa-ed96-4c98-8ce5-1419526d60e6" alt="image" width="150" height="400">
<img src="https://github.com/lferpaz/TFG/assets/57639021/109b0510-4ed0-4c3a-a61f-2391304b2947" alt="image" width="150" height="400">
<img src="https://github.com/lferpaz/TFG/assets/57639021/b88889d2-2813-4e9a-8c9f-e85abe2346f6" alt="image" width="150" height="400">




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

## Model metrics of interest ✔️
The results of the model were the following:

| Metric                            | Value   |
|-----------------------------------|---------|
| Precision                         | 0.927   |
| Positive Precision                | 0.935   |
| Sensitivity                       | 0.918   |
| F1-Score                          | 0.919   |
| Area under the ROC Curve (AUC)    | 0.927   |
| Average Precision                 | 0.870   |
| Average Sensitivity               | 0.680   |

The confusion matrix shows the performance of the model in classifying the samples. Here is the confusion matrix for the model:

|           | Predicted Authentic | Predicted Manipulated |
|-----------|---------------------|-----------------------|
| Authentic |         0.9          |          0.098           |
| Manipulated|         0.03         |          0.97           |

- The probability distribution shows the estimated probability for each class and the precision and recall curve of the model :
<img src="https://github.com/lferpaz/TFG/assets/57639021/77dea3eb-3171-434d-905e-b385ab65305b" alt="image" width="350" height="350">

<img src="https://github.com/lferpaz/TFG/assets/57639021/6824da5c-7214-410b-bf49-714e0b58be09" alt="image" width="350" height="350">

### Comparison with other models

| Model              | Precision | Year of Publication |
|--------------------|-----------|---------------------|
| Developed Model    |   93.5%   |       2023          |
| ModNet[16]         |   94.2%   |       2021          |
| DFI-Net [17]       |   92.3%   |       2019          |
| CAS-GAN [18]       |   91.8%   |       2020          |
| DCT-IF [19]        |   89.9%   |       2019          |
| TIDE[20]           |   87.6%   |       2019          |

## Results
<img src="https://github.com/lferpaz/TFG/assets/57639021/bea0066b-fc54-44cf-9aa7-db5848de8142" alt="image" width="350" height="150">
<img src="https://github.com/lferpaz/TFG/assets/57639021/937f970f-3f39-4273-9e7d-984fab273e97" alt="image" width="350" height="150">
<img src="https://github.com/lferpaz/TFG/assets/57639021/05d4fc76-6b00-4d4c-b64b-a9834ca1e920" alt="image" width="350" height="150">
<img src="https://github.com/lferpaz/TFG/assets/57639021/1b201f1a-f947-4389-ab99-08aee44349f6" alt="image" width="350" height="150">
<img src="https://github.com/lferpaz/TFG/assets/57639021/91337fdb-6cbd-4921-a446-f226886b2251" alt="image" width="350" height="150">
<img src="https://github.com/lferpaz/TFG/assets/57639021/1fadf901-eeed-4ea8-8f83-cd68e9ca2834" alt="image" width="350" height="150">
<img src="https://github.com/lferpaz/TFG/assets/57639021/d384e40e-dd1d-4026-b02f-afe8480c3f7b" alt="image" width="350" height="150">
<img src="https://github.com/lferpaz/TFG/assets/57639021/9da735fe-2913-454e-885b-1f2a844dfee0" alt="image" width="350" height="150">




## Contact :handshake:
If you have any questions or feedback about the project, please feel free to contact me at 1567369@uab.cat I am happy to discuss any aspect of the project with you.

## Acknowledgements :gem:

I would like to thank my supervisor Jordi Serra for her guidance and support throughout the project. I would also like to thank my colleagues for their valuable feedback and suggestions.

## License :warning:


