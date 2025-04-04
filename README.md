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
    * Loads image data from the specified directory structure.
    * Displays sample images from each class using `matplotlib.pyplot`.
    * Calculates and displays the shape of the images.
    * Arranges image display in a grid format, dynamically adjusting to the number of classes.
    * Includes error handling for missing image files or empty class directories.

2.  **Visualization:**
    * Displays all classes available in the dataset.
    * Dynamically calculates the number of rows and columns for the image grid.
    * Uses `matplotlib.pyplot` to display the images.
    * Displays image shapes and class names as titles.
    * Includes error handling for missing image files and empty class directories.
    * Uses `plt.tight_layout()` to prevent overlapping titles.
      
## Improvements

* **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and data augmentation techniques.
* **Model Architecture:** Try different pre-trained models or custom architectures.
* **Larger Dataset:** Use a larger dataset for better model performance.
* **Advanced Evaluation Metrics:** Implement more advanced evaluation metrics, such as precision-recall curves or ROC curves.
* **Save and load model:** Add code to save the trained model, and load it later for prediction.
* **Inference Code:** Add code to perform inference on new images.

## Author

[zaidaanshiraz]
