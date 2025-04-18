# 🐾 Multi-Class Animal Classification using Deep Learning

This project is a deep learning-based image classification system that can automatically recognize and classify images of animals into one of **90 categories** using **MobileNetV2** and **transfer learning** techniques.

---

## 📌 Problem Statement

Manually identifying animals in images is time-consuming and error-prone. This project solves that by building an AI-powered model that can accurately classify animal species based on their visual features, even when there are similarities between classes (e.g., cheetah vs. leopard).

---

## 🎯 Objectives

- Understand and implement multi-class image classification.
- Leverage transfer learning for faster and more accurate results.
- Train and fine-tune a deep CNN (MobileNetV2).
- Evaluate model performance with real-world metrics.
- Build a prediction pipeline for new unseen images.

---

## 🛠️ Tools & Technologies

- **Python 3**
- **TensorFlow / Keras**
- **MobileNetV2 (pretrained on ImageNet)**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **ImageDataGenerator**
- **Jupyter Notebook / Colab**
- **Kaggle Dataset**

---

## 🧠 Methodology

1. **Data Collection**: Downloaded a dataset of 90 animal classes from Kaggle.
2. **Preprocessing**: Resized, normalized, and augmented images using `ImageDataGenerator`.
3. **Model Building**: Used MobileNetV2 with custom dense layers for classification.
4. **Training**: Applied callbacks like EarlyStopping and ModelCheckpoint, followed by fine-tuning.
5. **Evaluation**: Visualized accuracy/loss, and analyzed results using confusion matrix and classification report.
6. **Deployment**: Created a prediction function to classify new animal images using the saved model.

---

## 📁 Dataset

- Dataset Source: [Kaggle - Animal Image Dataset (90 classes)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)
- Images are organized by folder names, where each folder is a separate class.

---

## 🧪 Results

- Achieved high classification accuracy across diverse animal species.
- Clear visualization of model performance using accuracy plots and confusion matrix.
- Trained model saved as `mcar.keras`.

---

## 🔍 Sample Prediction

```python
def predict_animal(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()
```

---

## 🚀 Future Improvements

- Integrate with a web app using Streamlit or Flask.
- Add Grad-CAM visualizations to explain model predictions.
- Handle multiple animals per image using object detection (e.g., YOLO or SSD).
- Improve dataset balance with augmentation or synthetic data generation.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Zaidaan Shiraz**  
[GitHub Profile](https://github.com/zaidaanshiraz)
