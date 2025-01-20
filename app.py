import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the pre-trained model
# loading in the model to predict on the data
pickle_in = open('XGBCTGAN.pkl', 'rb')
xgb = pickle.load(pickle_in)

# App Title with Image
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide", initial_sidebar_state="expanded",)

# Custom CSS for Modern Styling
st.markdown(
    """
    <style>
    /* Global Style */
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f4f4f9;
        color: #333333;
        font-size: 18px;
    }
    .main {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    /* Sidebar Styling */
    .stSidebar {
        background: linear-gradient(135deg, #FFD580, #FFAA5A); /* Darker orange gradient */
        color: black;/* Ensure contrast against darker background */
        border-radius: 15px;
        padding: 20px;
    }
    .stSidebar h2, .stSidebar h3 {
        color: black !important;
        font-size: 20px;
    }
    .stSidebar .stMarkdown {
        color: black !important;
        font-size: 18px;
    }
    .stSlider > div {
        color: black !important; /* Better text contrast on sliders */
        font-size: 16px;
    }
    .stButton>button {
        background-color: #FFA726;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 4px 6px rgba(0, 123, 255, 0.3);
    }
    .stButton>button:hover {
        background-color: #FF9800;
        transform: scale(1.05);
    }
    .stSlider .st-br {
        color: #0056b3 !important;
        font-size: 16px;
    }
    .title {
        font-family: 'Roboto', sans-serif;
        color: #007bff;
        font-size: 38px;
        font-weight: 600;
    }
    .subtitle {
        font-size: 20px;
        color: #333333;
        font-weight: 400;
    }
    /* Dataset Preview Card */
    .data-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: #888888;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section with Image
# Layout Adjustments
st.sidebar.markdown("<h2>📝 Input Health Details</h2>", unsafe_allow_html=True)

st.markdown("<div class='title'>🔍 Diabetes Prediction App</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Welcome to the Diabetes Prediction App 😊! Predict diabetes risk based on health indicators with AI assistance.</div>",
    unsafe_allow_html=True
)

# Dataset Preview
csv_path = "Diabetes_BRFSS2021_CTGANbalanced.csv"
df = pd.read_csv(csv_path)
st.markdown("<div class='data-card'>", unsafe_allow_html=True)
st.write("### Preview of Dataset")
st.dataframe(df.head(), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("## Predictions from the XGBoost Model")
st.write("This app employs a XGBoost model trained on a CTGAN-balanced diabetes dataset to predict diabetes.")

# Initialize user_input as a dictionary
user_input = {}

X = df.drop(columns=['Diabetes'])
y = df['Diabetes']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

categorical_columns = ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Vegetables',
                       'GenHealth', 'DiffWalk', 'Gender', 'Age', 'Education', 'HouseholdIncome']

descriptions = {
'HighBP': '0-No, 1-Yes',
'HighChol':'0-No, 1-Yes',
'Smoker': '0-No, 1-Yes',
'PhysActivity': '0-No, 1-Yes',
'Fruits': '0-No, 1-Yes',
'Vegetables': '0-No, 1-Yes',
'GenHealth': '1-Excellent, 2-Very_good, 3-Good, 4-Fair, 5-Poor',
'DiffWalk': '0-No, 1-Yes',
'Gender': '0- Female, 1- Male',
'Age': '1- 18-24, 2- 25-29, 3- 30-34, 4- 35-39, 5- 40-44, 6- 45-49, 7- 50-54, 8- 55-59, 9- 60-64, 10- 65-69, 11- 70-74, 12- 75-79, 13- ≥80',
'Education': '1-Never_attended_school_or_kindergarden, 2-Grades_1-8, 3-Grades_9-11, 4-Grade_12_or_GED, 5-College_1-3years, 6-College_≥4years',
'HouseholdIncome':'1-<$10k, 2-<$15k, 3-<$20k, 4-<$25k, 5-<$35k, 6-<$50k, 7-<$75k, 8-<$100k, 9-<$150k, 10-<$200k, 11-≥$200k',}


for column in X.columns:
    if column in categorical_columns:
        description = descriptions.get(column, '')
        sorted_unique_values = sorted(df[column].unique())
        user_input[column] = st.selectbox(f"{column}: {description}", sorted_unique_values)
    else:
        description = descriptions.get(column, '')

        if column == 'BMI':
            user_input[column] = st.sidebar.slider(
                f"{column} (Body Mass Index, Range: 12.0-99.3):",
                min_value=12.0,
                max_value=99.3,
                step=0.1
            )
        elif column == 'MentHealth':
            user_input[column] = st.sidebar.slider(
                f"{column} (Mental Health Days, Range: 0-30):",
                min_value=0,
                max_value=30,
                step=1
            )
        elif column == 'PhysHealth':
            user_input[column] = st.sidebar.slider(
                f"{column} (Physical Health Days, Range: 0-30):",
                min_value=0,
                max_value=30,
                step=1
            )
        else:
            # General case for other numerical variables
            user_input[column] = st.sidebar.slider(
                f"{column} (Range: {df[column].min()}-{df[column].max()}):",
                float(df[column].min()),
                float(df[column].max())
            )

# Convert user input to dataframe
user_input_df = pd.DataFrame([user_input])


# Prediction Logic
if st.button("🔮 Predict"):
    prediction = xgb.predict(user_input_df)
    prediction_proba = xgb.predict_proba(user_input_df)

    # Show Prediction
    st.markdown("### 💡📈 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️🤒 The model predicts that the person is **Diabetic**.")
    else:
        st.success("✅😊 The model predicts that the person is **Not Diabetic**.")

    # Show Prediction Probabilities
    st.markdown("### 📊 Prediction Probability")
    st.write(f"Not Diabetic: {prediction_proba[0][0] * 100:.2f}%")
    st.write(f"Diabetic: {prediction_proba[0][1] * 100:.2f}%")

# SHAP Explanation
st.subheader("SHAP Explanation of Prediction")
# Apply SHAP
explainer = shap.Explainer(xgb, X_train)
shap_values = explainer(user_input_df)

# Use SHAP's built-in waterfall plot for the first prediction

plt.rcParams.update({'font.size': 8})  # Reduce font size
fig, ax = plt.subplots(figsize=(5, 3))
shap.waterfall_plot(shap_values[0], max_display=10, show=False)  # SHAP values for the first instance
plt.gcf().set_dpi(100)  # Ensure consistent resolution

# Add text for explanation
plt.text(
    -0.2, 0.95,  # Position: adjust x and y to place it appropriately
    "Blue bars = Decrease the prediction\nRed bars = Increase the prediction",
    fontsize=10,
    color='black',
    transform=plt.gcf().transFigure,  # Use axis-relative coordinates
    bbox=dict(facecolor='white', alpha=0.8)  # Add a background box
)
# Adjust layout to prevent overlap
plt.tight_layout()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("<div class='footer'>📌 Note: This is a prediction tool and not a medical diagnosis. Consult a healthcare provider 👩🏻‍⚕️ for professional advice.</div>", unsafe_allow_html=True)
st.markdown("👨‍💻 Developed by [Alicia]")

