import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm, linear_model, ensemble, neighbors
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# Load and preprocess data
def load_data():
    df = pd.read_csv('diabetes.csv')
    return df

def preprocess_data(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, y, X_scaled, scaler

df = load_data()
X, y, X_scaled, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Define models
models = {
    'SVM': svm.SVC(kernel='linear', probability=True),
    'Linear Regression': linear_model.LogisticRegression(),
    'Random Forest': ensemble.RandomForestClassifier(),
    'KNN': neighbors.KNeighborsClassifier()
}

# Train models
for name, model in models.items():
    model.fit(X_train, y_train)

# Define functions for plotting
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def plot_roc_curve(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='cyan', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)

def plot_data_summary(df):
    st.subheader('Data Summary')
    st.write(df.describe())
    
    st.subheader('Feature Distributions')
    num_features = df.shape[1] - 1
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 18))
    axes = axes.flatten()
    
    for i, feature in enumerate(df.columns[:-1]):
        ax = axes[i]
        sns.histplot(df[feature], kde=True, bins=30, ax=ax, color='cyan', edgecolor='black')
        ax.set_title(f'Distribution of {feature}')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout()
    st.pyplot(fig)

def provide_health_recommendations(prediction):
    if prediction == 1:
        # Diabetic Recommendations
        st.subheader("Health Recommendations for Diabetic Patients")
        st.markdown("""
        - **Dietary changes**: Focus on consuming whole grains, vegetables, lean proteins, and healthy fats. Reduce intake of processed foods, sugary snacks, and drinks.
        - **Physical activity**: Aim for at least 30 minutes of moderate-intensity exercise (e.g., walking, swimming) five days a week.
        - **Blood sugar monitoring**: Regularly monitor your blood glucose levels and maintain a record to share with your healthcare provider.
        - **Medication adherence**: Take medications as prescribed by your doctor. Discuss any side effects or concerns with your healthcare provider.
        - **Regular check-ups**: Schedule frequent check-ups to monitor blood sugar levels, kidney function, and cardiovascular health.
        - **Stress management**: Engage in stress-relieving activities like meditation, yoga, or hobbies to maintain emotional well-being.
        """)
    else:
        # Non-Diabetic Recommendations
        st.subheader("Health Recommendations for Non-Diabetic Individuals")
        st.markdown("""
        - **Healthy eating**: Maintain a balanced diet rich in vegetables, whole grains, and healthy fats. Avoid excessive sugar and processed foods.
        - **Regular physical activity**: Engage in at least 150 minutes of moderate-intensity exercise per week (e.g., jogging, biking).
        - **Weight management**: Keep a healthy weight through a balanced diet and physical activity.
        - **Regular health screening**: Even without diabetes, consider regular check-ups, especially if you have risk factors (family history, obesity).
        - **Stay hydrated**: Drink plenty of water throughout the day to support overall health.
        - **Avoid smoking and limit alcohol**: These habits can increase the risk of developing chronic conditions, including diabetes.
        """)

from sklearn.inspection import permutation_importance

def model_tabs(model_name):
    st.title(f'{model_name} Model Analysis')
    
    # Create tabs for sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Model Description', 'Data Description', 'Detection', 'Feature Importance', 'Confusion/ROC'])

# Model Description Tab
    # Model Description Tab
    with tab1:
        st.subheader('Model Description')
        if model_name == 'SVM':
            st.write(f"""
            ### Support Vector Machine (SVM) Model

            **Overview:**
            This app uses a Support Vector Machine (SVM) with a linear kernel to predict whether someone is likely to have diabetes based on health measurements. SVM is renowned for its precision and effectiveness in classification tasks.

            **What is SVM?**
            A Support Vector Machine (SVM) is a sophisticated classification algorithm that works by finding the optimal hyperplane that maximizes the margin between different classes. Think of it as drawing the best line (or plane) to separate different categories.

            **How Does it Work?**
            - **Linear Kernel:** The SVM used here applies a linear kernel, suitable for problems where classes can be separated with a straight line.
            - **Support Vectors:** Key data points closest to the hyperplane that determine its position. They are like the "border patrol" for classification!

            **Fun Fact:**
            SVMs were inspired by the concept of separating hyperplanes in geometry. The idea is to find the best possible divider between classes, much like finding the perfect line to split two groups of colored dots!

            **Insights:**
            SVMs are especially powerful in high-dimensional spaces and are often used in applications requiring high accuracy, such as image recognition and text classification.

            **Why Use SVM?**
            - **Precision:** SVMs can be very accurate with well-chosen features.
            - **Flexibility:** They handle various types of data well, thanks to different kernels and hyperplane choices.

            """)

        elif model_name == 'Linear Regression':
            st.write(f"""
            ### Logistic Regression Model

            **Overview:**
            This app uses a Logistic Regression model to predict whether someone might have diabetes based on their health data. Despite its name, logistic regression is primarily used for classification tasks.

            **What is Logistic Regression?**
            Logistic Regression is a statistical method for binary classification. It estimates the probability of an outcome by applying the logistic function, which outputs a value between 0 and 1.

            **How Does it Work?**
            - **Logistic Function:** It transforms the output of a linear equation into a probability. The function is shaped like an "S" curve, making it ideal for binary outcomes.
            - **Coefficients:** The model uses weights assigned to each feature to make predictions.

            **Fun Fact:**
            Logistic Regression was named for the logistic function, which itself is named after "logistic growth" in population studies. It’s a bit like applying a growth curve to predict categories!

            **Insights:**
            Logistic Regression is valued for its simplicity and interpretability, making it a great starting point for binary classification tasks.

            **Why Use Logistic Regression?**
            - **Clarity:** Provides clear insights into feature importance through its coefficients.
            - **Efficiency:** Quick to train and often performs well with straightforward problems.

            """)

        elif model_name == 'Random Forest':
            st.write(f"""
            ### Random Forest Model

            **Overview:**
            This app uses a Random Forest Classifier to predict diabetes by combining multiple decision trees. Random Forest is known for its robustness and accuracy.

            **What is Random Forest?**
            Random Forest is an ensemble learning technique that builds a collection of decision trees. Each tree is trained on a different subset of the data, and their predictions are aggregated to produce a final result.

            **How Does it Work?**
            - **Decision Trees:** Each tree in the forest makes its own prediction based on the data subset it has.
            - **Bagging:** This technique trains each tree on a random sample of the data, reducing the risk of overfitting.

            **Fun Fact:**
            Random Forest is like having a team of decision trees working together. Each tree brings its own perspective, and their collective judgment often leads to more accurate predictions!

            **Insights:**
            Random Forest excels in handling large datasets and complex data structures, making it a favorite in various machine learning competitions.

            **Why Use Random Forest?**
            - **Accuracy:** Generally provides high accuracy by averaging multiple trees.
            - **Versatility:** Handles different types of data and can model complex interactions.

            """)

        elif model_name == 'KNN':
            st.write(f"""
            ### K-Nearest Neighbors (KNN) Model

            **Overview:**
            This app uses a K-Nearest Neighbors (KNN) model to classify whether someone is likely to have diabetes based on health measurements. KNN is a straightforward and intuitive algorithm.

            **What is KNN?**
            K-Nearest Neighbors (KNN) is a classification algorithm that makes predictions based on the majority class among the nearest neighbors of a data point. It’s like asking your closest friends for their opinion on a question!

            **How Does it Work?**
            - **Distance Metric:** KNN calculates the distance between data points using methods like Euclidean distance. It finds the "K" nearest neighbors and determines the class by majority vote.
            - **Voting Mechanism:** The class that appears most frequently among the neighbors is chosen as the prediction.

            **Fun Fact:**
            KNN can be seen as a "popularity contest" where the most common class among neighbors wins. Imagine a neighborhood vote where the most popular choice prevails!

            **Insights:**
            KNN is easy to understand and implement but can be slow for large datasets because it needs to calculate distances for every prediction.

            **Why Use KNN?**
            - **Simplicity:** Very intuitive and easy to implement.
            - **Adaptability:** No explicit training phase, making it adaptable to various data scenarios.

            """)

    # Data Description Tab
    with tab2:
        st.subheader('Data Description')
        plot_data_summary(df)
    
    # Detection Tab
    with tab3:
        st.subheader('Detection')
        intro_text = st.empty()
        intro_text.write("""
        ### Detection and Health Recommendations
        Enter your health metrics in the sidebar to get a prediction and health recommendations.
        """)
        
        st.sidebar.title('Input Features')
        with st.sidebar.container():
            st.subheader('Adjust the settings below:')
            preg = st.slider('Pregnancies', 0, 17, 3, 1)
            glucose = st.slider('Glucose', 0, 199, 117, 1)
            bp = st.slider('Blood Pressure', 0, 122, 72, 1)
            skinthickness = st.slider('Skin Thickness', 0, 99, 23, 1)
            insulin = st.slider('Insulin', 0, 846, 30, 1)
            bmi = st.slider('BMI', 0.0, 67.1, 32.0, 0.1)
            dpf = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
            age = st.slider('Age', 21, 81, 29, 1)
        
        if st.sidebar.button('Predict'):
            user_input = np.array([[preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]])
            user_input = scaler.transform(user_input)
            prediction = models[model_name].predict(user_input)[0]
            intro_text.empty()
            # Prediction output with emojis
            if prediction == 1:  # Diabetic
                st.write(f'**🩺 Prediction:** You are **Diabetic**. 😷')
            else:  # Non-Diabetic
                st.write(f'**🩺 Prediction:** You are **Non-Diabetic**. 😊')
                        # Provide health recommendations based on prediction
            provide_health_recommendations(prediction)

    # Feature Importance Tab
    with tab4:
        st.subheader('Feature Importance')
        if model_name == 'Random Forest':
            feature_importances = models[model_name].feature_importances_
            features = df.columns[:-1]  # Exclude the target variable
            importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            st.write(importance_df)
            st.bar_chart(importance_df.set_index('Feature'))
        elif model_name == 'Linear Regression':
            # Compute permutation importance
            results = permutation_importance(models[model_name], X_test, y_test, n_repeats=10, random_state=0, scoring='accuracy')
            importance_df = pd.DataFrame({'Feature': df.columns[:-1], 'Importance': results.importances_mean})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            st.write(importance_df)
            st.bar_chart(importance_df.set_index('Feature'))
        else:
            st.write(f'{model_name} does not provide feature importance.')

    # Confusion/ROC Tab
    with tab5:
        st.subheader('Confusion Matrix and ROC Curve')
        plot_confusion_matrix(models[model_name], X_test, y_test)
        plot_roc_curve(models[model_name], X_test, y_test)

# Define the app
def app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "SVM Model", "Linear Regression Model", "Random Forest Model", "KNN Model"])

    if page == "Home":
        st.title("Welcome to the Diabetes Prediction App")
        
        # Introduction
        st.write("""
       
                 
        ### Get Started
        Select a model from the sidebar to start exploring. Each model offers unique insights and predictive capabilities.

        ### How It Works
        - **Select a Model:** Choose from SVM, Logistic Regression, Random Forest, or KNN.
        - **Input Health Data:** Enter your health measurements such as glucose levels, BMI, and age.
        - **Get Prediction:** See whether you are classified as "Diabetic" or "Non-Diabetic" based on the model's analysis.

        ### What Can You Learn?
        - **Model Insights:** Understand how each model makes predictions and which features are important.
        - **Model Comparisons:** Compare the performance and accuracy of different models.
        - **Health Metrics:** Learn about the significance of various health metrics in diabetes prediction.

        ### Features of This App
        - **Interactive Predictions:** Input your health data to receive real-time predictions.
        - **Model Descriptions:** Detailed explanations of each machine learning model used in the app.
        - **Performance Insights:** Overview of how each model is evaluated and performs.



        """)
        
        # Add an image or infographic (if available)
        # st.image("path_to_image.jpg", caption="Diabetes Prediction Models")

    elif page == "SVM Model":
        model_tabs('SVM')
    elif page == "Linear Regression Model":
        model_tabs('Linear Regression')
    elif page == "Random Forest Model":
        model_tabs('Random Forest')
    elif page == "KNN Model":
        model_tabs('KNN')


if __name__ == "__main__":
    app()



