import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="My Personal Diabetes Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body { background-color: #f7f7f7; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    .stButton button {
        background-color: #008080; color: white; border: none; padding: 10px 20px;
        font-size: 15px; border-radius: 5px; transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #f7f7f7; color: #008080; border: 1px solid #008080;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("Navigation")
st.sidebar.markdown("Hi, I'm Salsa! Welcome to my project.")
page = st.sidebar.radio("Choose a page:", ["Home", "Data Analysis", "Model Comparison", "Predict Diabetes", "About Me"])


@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
    df.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
    return df

@st.cache_resource
def train_models(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_model = LogisticRegression()
    log_model.fit(X_train_scaled, y_train)
    log_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
    
    return scaler, log_model, rf_model, log_acc, rf_acc

df = load_data()
scaler, log_model, rf_model, log_acc, rf_acc = train_models(df)

if page == "Home":
    st.title("Welcome to My Diabetes Predictor!")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("https://apolloclinicguwahati.com/wp-content/uploads/2020/07/diabetes.png", use_container_width=True)
    st.markdown("""
    Hello there! This is my personal project where I explore how machine learning can help predict the risk of diabetes.
    
    In this app, you can:
    - **Analyze Data:** See how different health indicators correlate.
    - **Compare Models:** Check how two different models perform.
    - **Make Predictions:** Enter patient data and get a quick prediction.
    
    Use the sidebar to navigate.
    """)

elif page == "Data Analysis":
    st.title("Data Analysis")
    st.markdown("Let's dive into the diabetes dataset and uncover some insights.")
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    st.markdown("### Summary Statistics")
    st.write(df.describe())

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Comparing two models for diabetes prediction.")
    st.markdown("#### Model Performance")
    st.write(f"**Logistic Regression Accuracy:** {log_acc:.2f}")
    st.write(f"**Random Forest Accuracy:** {rf_acc:.2f}")
    acc_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [log_acc, rf_acc]
    })
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Model", y="Accuracy", data=acc_data, palette="mako", ax=ax2)
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

elif page == "Predict Diabetes":
    st.title("Predict Diabetes")
    st.markdown("Enter patient data and get a prediction.")
    with st.form("my_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", 0, 20, value=1)
            glucose = st.number_input("Glucose Level", 0, 200, value=130)
            blood_pressure = st.number_input("Blood Pressure", 0, 150, value=70)
            skin_thickness = st.number_input("Skin Thickness", 0, 100, value=20)
        with col2:
            insulin = st.number_input("Insulin", 0, 500, value=80)
            bmi = st.number_input("BMI", 0.0, 50.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree", 0.0, 2.5, value=0.5, step=0.01)
            age = st.number_input("Age", 0, 120, value=30)
        model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest"])
        submit = st.form_submit_button("Predict Now")
    
    if submit:
        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        user_input_scaled = scaler.transform(user_input)
        with st.spinner("Crunching numbers..."):
            time.sleep(1)
            if model_choice == "Logistic Regression":
                prediction = log_model.predict(user_input_scaled)
            else:
                prediction = rf_model.predict(user_input_scaled)
        if prediction[0] == 1:
            st.error("🚨 The prediction suggests that the patient is likely to have diabetes.")
        else:
            st.success("✅ The prediction suggests that the patient is unlikely to have diabetes.")

elif page == "About Me":
    st.markdown(
        """
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.8); }
            to { opacity: 1; transform: scale(1); }
        }
        .animated-image {
            animation: fadeIn 1.5s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    import time  
    timestamp = int(time.time())  # Menghindari cache dengan timestamp

    # Deklarasikan image_url SEBELUM digunakan
    image_url = f"https://github.com/SalsabilaLubis21/MyPersonalDiabetesPredictor/blob/main/linkedin%20photo.jpg?raw=true&t={timestamp}"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        time.sleep(0.5)  

        # Buat container kosong
        img_container = st.empty()
        
        # Pastikan image_url sudah dideklarasikan sebelum digunakan di sini
        img_container.markdown(f'<img src="{image_url}" class="animated-image" width="450">', unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.title("About Me")
    st.markdown("""
    **Hi, I'm Salsa!**  
    
    I'm a first year student at President University and passionate about data science, machine learning,and interactive web applications. This project shows how machine learning can be applied to diabetes prediction.
    
    **Technologies Used:**  
    - Python, Pandas, NumPy  
    - Scikit-learn  
    - Streamlit  
    - Matplotlib and Seaborn
    
    Check out my [GitHub repository](https://github.com/SalsabilaLubis21) for more projects.
    """)
