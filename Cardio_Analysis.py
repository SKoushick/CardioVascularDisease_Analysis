import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from PIL import Image
import joblib as jl

# Inject CSS for custom styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #003366;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #004080;
        color: white;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #0059b3;
        color: #ffffff;
        transform: scale(1.03);
    }
    .sidebar .sidebar-content {
        background-color: #d9e2f3;
        padding: 20px 10px;
        border-radius: 8px;
    }
    .css-1v3fvcr, .stDataFrame, .css-1y4p8pa {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
    }
    .stMarkdown, .css-1cpxqw2 {
        font-size: 17px;
        line-height: 1.6;
    }
    .stImage > img {
        border-radius: 10px;
        border: 3px solid #cccccc;
    }
    .prediction-container {
        background-color: #eaf2fb;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .prediction-container label, .prediction-container select, .prediction-container input {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_data():
    df = pd.read_csv("cardio-train-edited.csv")
    return df

@st.cache_resource
def load_model():
    model = jl.load("Prediction_model.pkl")
    return model

# Load data and model
df = load_data()
model = load_model()

# Sidebar navigation buttons as individual buttons
st.sidebar.title("🩺 Cardiovascular App Navigation")
nav_buttons = {
    "Home": "Home",
    "Prediction": "Prediction",
    "Visualizations": "Visualizations",
    "Data Overview": "Data Overview",
    "About": "About"
}

if "page" not in st.session_state:
    st.session_state.page = "Home"

for label, value in nav_buttons.items():
    if st.sidebar.button(label):
        st.session_state.page = value

page = st.session_state.page

# Page logic
if page == "Home":
    st.title('Cardiovascular Disease')
    st.header("What is Cardiovascular Disease?")
    st.write("""
    Cardiovascular disease (CVD) is a general term for conditions affecting the heart or blood vessels.
    It's usually associated with a build-up of fatty deposits inside the arteries (atherosclerosis) and an increased risk of blood clots.
    It can also be associated with damage to arteries in organs such as the brain, heart, kidneys and eyes.
    """)
    img = Image.open("Edited.jpg")
    resized_img = img.resize((500, 300))
    st.image(resized_img, caption='Cardiovascular Disease')
    st.header("Causes of Cardiovascular Disease")
    st.write("""
    The most common causes behind cardiovascular diseases include:
    - High blood pressure
    - Smoking
    - High cholesterol
    - Diabetes
    - Inactivity
    - Being overweight or obese
    - Family history of CVD
    - Ethnic background
    - Excessive alcohol intake
    """)
    st.header("Symptoms")
    st.write("""
    If these symptoms are present, the person may be having a cardiovascular event:
    - Chest pain
    - Pain, weakness or numbness in the limbs
    - Shortness of breath
    - Very fast or slow heartbeat
    - Dizziness or fainting
    - Fatigue
    - Swelling in the legs, ankles or feet
    """)
    st.header('Treatment')
    st.write("""
    The best way to prevent cardiovascular disease is by adopting a healthy lifestyle:
    - Stop smoking
    - Eat a balanced diet
    - Exercise regularly
    - Maintain a healthy weight
    - Reduce alcohol consumption
    - Manage stress
    Medication and sometimes surgery may also be required, depending on the severity of the condition.
    """)

elif page == "Prediction":
    st.title('Prediction')
    Age = st.number_input('Age', min_value=0 ,max_value=100)
    Gender = st.selectbox('Gender',['1','2'])
    st.info('the value 1 is women and value 2 is men')
    Height = st.number_input('Height (cm)', min_value=0, max_value=300)
    Weight = st.number_input('Weight (kg)', min_value=0, max_value=300)
    Systolic_BP = st.number_input('Systolic Blood Pressure (mmHg)', min_value=0, max_value=300)
    Diastolic_BP = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=0, max_value=300)
    Cholesterol = st.selectbox('Cholesterol Level', ['1', '2', '3'])
    st.info("""Cholesterol levels: 
- 1 - Normal.
- 2 - Above Normal.
- 3 - Well Above Normal. 
            """)
    Glucose = st.selectbox('Glucose Level', ['1', '2', '3'])
    st.info("""Glucose levels:
- 1 - Normal.
- 2 - Above Normal.
- 3 - Well Above Normal.
            """)
    Alcohol = st.selectbox('Alcohol Intake', ['1', '2'])
    st.info("""Alcohol intake:
- 1 - No. 
- 2 - Yes.
            """)
    Activity = st.selectbox('Activity Level', ['1', '2'])
    st.info("""Activity level:
- 1 - Low. 
- 2 - High.
            """)

    click = st.button('Predict')
    if click:
    # Ensure the order of features matches the training data
        input_data = np.array([[Age, Gender, Height, Weight, Systolic_BP, Diastolic_BP, Cholesterol, Glucose, Alcohol, Activity]])

    # Predict using the GridSearchCV or Pipeline model
        prediction = model.predict(input_data)
        st.write("Input shape:", input_data.shape)
        st.write("Prediction:", prediction)


        st.header("Predicted Result")
        if prediction[0] == 0:
            st.success("No possibility of heart attack detected.")
        else:
            st.error("Future heart attack detected. Please consult a doctor for further advice.")

elif page == "Visualizations":
    st.title('Visualizations')
    st.header("Analysis")
    st.write("You can visualize the data to gain insights about the patterns and distribution of various health indicators among individuals.")
    st.header("Correlation Heatmap")
    st.image("output.png")
    st.header("Cholesterol Level Distribution")
    st.image("output1.png")
    st.header("Activity Level Distribution")
    st.image("output2.png")
    st.header("Glucose Level Distribution")
    st.image("output3.png")
    st.header("Cholesterol Level Distribution Over Age")
    st.image("output4.png")

elif page == "Data Overview":
    st.header('Data Overview')
    st.dataframe(df)

elif page == "About":
    st.header('About This Project')
    st.subheader("How soon after treatment will I feel better?")
    st.write("""
After you’ve had a heart attack, you’re at a higher risk of a similar occurrence. Your healthcare provider will likely recommend follow-up monitoring, testing and care to avoid future heart attacks. Some of these include:

- Heart scans: Similar to the methods used to diagnose a heart attack, these can assess the effects of your heart attack and determine if you have permanent heart damage. They can also look for signs of heart and circulatory problems that increase the chance of future heart attacks.
- Stress test: These heart tests and scans that take place while you’re exercising can show potential problems that stand out only when your heart is working harder.
- Cardiac rehabilitation: These programs help you improve your overall health and lifestyle, which can prevent another heart attack.


Additionally, you’ll continue to take medicines — some of the ones you received for immediate treatment of your heart attack — long-term. These include:

- Beta-blockers.
- ACE inhibitors.
- Aspirin and other blood-thinning agents.""")

    st.subheader("How soon after treatment will I feel better?")
    st.write("""
In general, your heart attack symptoms should decrease as you receive treatment. You’ll likely have some lingering weakness and fatigue during your hospital stay and for several days after. Your healthcare provider will give you guidance on rest, medications to take, etc.

Recovery from the treatments also varies, depending on the method of treatment. The average hospital stay for a heart attack is between four and five days. In general, expect to stay in the hospital for the following length of time:

- Medication only: People treated with medication only have an average hospital stay of approximately six days.
- PCI: Recovering from PCI is easier than surgery because it’s a less invasive method for treating a heart attack. The average length of stay for PCI is about four days.
- CABG: Recovery from heart bypass surgery takes longer because it’s a major surgery. The average length of stay for CABG is about seven days.
    """)
