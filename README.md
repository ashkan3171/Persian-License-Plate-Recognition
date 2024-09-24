# Persian License Plate Recognition

This project focuses on the detection and classification of Persian characters and numbers from car license plates using computer vision and machine learning techniques.

## Project Overview

The goal of this project is to identify car license plates and extract the text from the plates using image processing methods. The process involves several key steps:

1. Preprocessing the input image to detect the car and locate the license plate.
2. Segmenting the license plate into its individual characters.
3. Extracting relevant features from the characters.
4. Classifying the characters to identify the numbers and letters.

## Key Features

- **License Plate Detection**: Detects the location of a license plate in an image using contour detection based on various geometric properties.
- **Character Segmentation**: Segments characters on the license plate using histogram-based and thresholding methods.
- **Feature Extraction**: Extracts features from segmented characters using methods such as LBP (Local Binary Patterns), Sobel edge detection, SIFT, and more.
- **Classification**: Classifies characters using various machine learning models like Random Forest (RF), SVM (Support Vector Machine), MLP (Multi-Layer Perceptron), KNN (K-Nearest Neighbors), and Decision Tree (DT).
- **Evaluation**: Includes accuracy evaluation of various methods for segmentation and classification.

## Installation

To use this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/persian-license-plate-recognition.git
cd persian-license-plate-recognition
pip install -r requirements.txt
```

Usage
-----

1. Prepare your input images in the `data` directory.
2. Run the license plate detection:

bash

Copy code

`python detect_plate.py --input data/car_images`

3. Segment the detected license plates:

bash

Copy code

`python segment_plate.py --input data/detected_plates`

4. Extract features and classify the characters:

bash

Copy code
`python classify_characters.py --input data/segmented_characters`
-------

Dataset
-------

The project uses a set of Persian license plate images for training and testing the model. You can use your own dataset by placing images in the `data/car_images` directory.
Results

-------

The final output of the program will be the license plate numbers printed from the input image. Example output:

javascript

Copy code

`License Plate Number: 63 ن 10-545`
Methods Used

------------

* **Detection**: Contour-based method for plate detection.
* **Segmentation**: Histogram-based and thresholding methods for character segmentation.
* **Feature Extraction**:
  * **LBP (Local Binary Pattern)**
  * **SIFT (Scale-Invariant Feature Transform)**
  * **Sobel Edge Detection**
  * **Corner Detection (Harris, Shi-Tomasi)**
* **Classification**: Random Forest, SVM, MLP, KNN, Decision Tree.

Evaluation
----------

The methods were evaluated using precision, recall, and accuracy metrics. See the detailed evaluation results in the report.
Future Work

-----------

* Improve character recognition accuracy using deep learning methods.
* Expand the system to work with license plates from other countries.
* Implement real-time detection using video input.

Contributors
------------

* Ashkan Sheikhansari
* Abolfazl Abedini
1. 
