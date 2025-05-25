# 🌸 Iris Species Classifier
A simple but elegant machine learning project that classifies iris flowers using Support Vector Machines (SVM), Principal Component Analysis (PCA), and a Streamlit app.

# 🔍 Project Overview
In this project, I built a classifier to predict the species of an iris flower based on its petal and sepal dimensions. I used a tuned Support Vector Machine (SVM) wrapped in a pipeline and visualized the predictions using PCA-reduced components and Gaussian Mixture Models (GMM). The final result is a lightweight interactive web app powered by Streamlit.

# 📁 Files in this repo
| File Name       | Description |
|----------------|-------------|
| **Iris.csv** | Dataset containing iris species data |
| **iris_classifier.ipynb** | Jupyter notebook used for data exploration, model training, and evaluation |
| **requirements.yaml** | Conda environment file for reproducibility |
| **app.py** | The main Streamlit application |
| **util.py** | Holds reusable functions for generating model performance visualizations |



Iris.csv: The classic Iris dataset from UCI.

iris_svm_pipeline.joblib: Pre-trained SVM model with scaling and tuning.

label_encoder.joblib: Label encoder used to transform species labels.

iris_app.py: The main Streamlit application.

iris_classifier.ipynb: Jupyter notebook used for data exploration, model training, and evaluation.

requirements.yaml: Conda environment file for reproducibility.

pca_gmm_iris.jpg: Custom visualization of GMM density over PCA space.

# 🧠 Model Performance
The final model achieved ~97% accuracy on both training and validation sets, indicating good generalization and no signs of overfitting.
I used GridSearchCV to tune the C, gamma, and kernel parameters of the SVM.

# 🌐 Web App
You can interact with the model using the Streamlit app. Users can input measurements via sliders and instantly see the predicted species. A dynamic plot visualizes where your flower lands in PCA space with GMM density contours and an annotated prediction arrow.

# 📌 Key Features
End-to-end pipeline: preprocessing + SVM

PCA for dimensionality reduction

GMM for visualizing density in feature space

Live visualization with annotation in Streamlit

Caches and modular functions for performance
