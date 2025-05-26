# Import packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import joblib

# Set app title
st.title("🌸 :blue[Iris Flower Species Classifier]")

# Load cached model and encoder
@st.cache_resource
def load_model():
    return joblib.load("Model/iris_svm_pipeline.joblib")

@st.cache_resource
def load_encoder():
    return joblib.load("Model/label_encoder.joblib")

model = load_model()
label_encoder = load_encoder()

# Load data
df = pd.read_csv("Iris.csv",
                 usecols=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"])
X_train = df.drop("Species", axis=1)
y_train = df["Species"]

# Create sidebar input for measurements
st.sidebar.header("🌿 Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.2)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Predict species from measurement
def get_prediction(sepal_length, sepal_width, petal_length, petal_width):
    user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    prediction = model.predict(user_input)[0]
    return prediction, user_input

# GMM + PCA Plot
def plot_gmm_with_prediction(X_train, y_train, model, new_sample_df, predicted_class):

    # Apply PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    new_sample_pca = pca.transform(new_sample_df)

    # GMM Fit
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X_train_pca)

    # GMM Contours
    x = np.linspace(X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1, 300)
    y = np.linspace(X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1, 300)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid = np.c_[X_grid.ravel(), Y_grid.ravel()]
    Z = np.exp(gmm.score_samples(grid)).reshape(X_grid.shape)

    # Prepare DataFrame for plotting
    pca_df = pd.DataFrame(X_train_pca, columns=["PC1", "PC2"])
    pca_df["Species"] = y_train

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Species", palette="Set1", ax=ax, s=50)
    ax.contour(X_grid, Y_grid, Z, levels=15, cmap='Purples', alpha=0.5)

    # Plot measurement
    ax.scatter(new_sample_pca[0, 0], new_sample_pca[0, 1], c="black", s=120, edgecolors="white", label="New Sample")

    # Annotate prediction
    ax.annotate(f"Prediction: {predicted_class}",
                xy=(new_sample_pca[0, 0], new_sample_pca[0, 1]),
                xytext=(new_sample_pca[0, 0] + 1.0, new_sample_pca[0, 1] + 0.5),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=12, color='black')

    ax.set_title("PCA + GMM Density with Prediction")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(title="Species")
    return fig

# Predict and plot measurement when user classifies
if st.button("🔍 Classify"):
    # Get prediction and measurement
    predicted_species, user_input_df = get_prediction(sepal_length, sepal_width, petal_length, petal_width)

    # Display prediction
    st.success(f"🌼 The predicted Iris species is: **{predicted_species}**")

    # Plot prediction measurements on PCA map
    fig = plot_gmm_with_prediction(X_train, y_train, model, user_input_df, predicted_species)
    st.pyplot(fig)
