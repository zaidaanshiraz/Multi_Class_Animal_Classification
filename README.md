# Multi-Class Animal Image Classification

This Jupyter Notebook (`multi_class_animal_classification.ipynb`) demonstrates how to build and train a deep learning model for multi-class animal image classification. It utilizes the MobileNetV2 architecture, a pre-trained convolutional neural network, for efficient feature extraction and fine-tuning.

## Dataset

The dataset used in this notebook is the "Animal Image Dataset (90 Different Animals)" from Kaggle, available at: [https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

It contains images of 90 different animal species, organized into separate directories for each class.

## Project Structure

* `multi_class_animal_classification.ipynb`: The main Jupyter Notebook containing the code for data loading, preprocessing, model building, training, and evaluation.

## Dependencies

* Python 3.x
* TensorFlow (with Keras)
* NumPy
* Matplotlib
* Scikit-learn
* Kaggle API (for downloading the dataset, if needed)

## Installation

1.  **Install Dependencies:**

    ```bash
    pip install tensorflow numpy matplotlib scikit-learn kaggle
    ```

2.  **Download the Dataset (if not already downloaded):**

    * If you are running this notebook in Kaggle, the dataset will be automatically downloaded.
    * If you are running it locally, you need to download the dataset from Kaggle and place it in the appropriate directory or use the kaggle API within the notebook.

3.  **Run the Notebook:**

    * Open `multi_class_animal_classification.ipynb` in Jupyter Notebook or JupyterLab.
    * Execute the cells sequentially.

## Code Overview

1.  **Data Loading and Preprocessing:**
    * Uses `ImageDataGenerator` for efficient batch processing, data augmentation, and train/validation splitting.
    * Resizes images to a consistent size for MobileNetV2 input.
2.  **Model Building:**
    * Loads the MobileNetV2 pre-trained model (weights from ImageNet).
    * Adds custom layers for classification (GlobalAveragePooling, Dense, Dropout, Softmax).
    * Freezes the base MobileNetV2 layers for fine-tuning.
3.  **Model Training:**
    * Compiles the model with Adam optimizer and categorical cross-entropy loss.
    * Uses `EarlyStopping` and `LearningRateScheduler` callbacks for better training.
    * Trains the model using the training data generator.
4.  **Model Evaluation:**
    * Evaluates the model on the validation data.
    * Generates a classification report and confusion matrix.
    * Visualizes sample images from the dataset.

## Improvements

* **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and data augmentation techniques.
* **Model Architecture:** Try different pre-trained models or custom architectures.
* **Larger Dataset:** Use a larger dataset for better model performance.
* **Advanced Evaluation Metrics:** Implement more advanced evaluation metrics, such as precision-recall curves or ROC curves.
* **Save and load model:** Add code to save the trained model, and load it later for prediction.
* **Inference Code:** Add code to perform inference on new images.

## Author

[zaidaanshiraz]
