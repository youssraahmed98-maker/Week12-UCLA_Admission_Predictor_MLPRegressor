# UCLA Admission Predictor with Neural Networks
## Project Overview
This project uses a **Neural Network (MLPRegressor)** to predict students’ admission probability based on their academic profile, including GRE score, TOEFL score, CGPA, and research experience.

The app provides an interactive experience where users can input their own information and receive their predicted admission probability, compare it to typical ranges, and visually explore relationships within the dataset.

---

## 👉 Live App:
https://ucla-admission-predictor-mlpregresor.streamlit.app/

---

## 🚀 Features
Interactive Prediction: Users input GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research experience to get their admission probability.

Prediction Output: Displays the predicted admission chance as a percentage.

Interpretation: Groups results into Low, Moderate, or High likelihood for easier understanding.

Visualizations: Shows relationships between features using multiple charts.

Neural Network Insights: Displays the model loss curve for training performance.

Error Handling: Handles missing model or data files gracefully.

---

## 📦 Dataset
The dataset used for training is `Admission.csv`, containing the following features:

Column Name	Description
GRE_Score	GRE score (out of 340)
TOEFL_Score	TOEFL score (out of 120)
University_Rating	University rating (1–5)
SOP	Statement of Purpose strength (1–5)
LOR	Letter of Recommendation strength (1–5)
CGPA	GPA (out of 10)
Research	Research experience (0 = No, 1 = Yes)
Admit_Chance	Probability of admission (target variable)

---

## 🛠 Technologies Used
Python 3.x  

## 📚 Libraries:
pandas: Data handling and preprocessing  
scikit-learn: Neural Network model (MLPRegressor)  
matplotlib, seaborn: Visualization  
pickle: Saving/loading trained model  
Streamlit: Web app interface  
numpy: Numerical operations  

---

## 🔍 Code Explanation
load_and_preprocess_data(): Loads the dataset and handles missing values. :contentReference[oaicite:0]{index=0}  

build_features(): Splits dataset into input features (X) and target variable (Admit_Chance). :contentReference[oaicite:1]{index=1}  

train_NNmodel(): Trains the Neural Network (MLPRegressor), applies MinMax scaling, and saves the model and scaler. :contentReference[oaicite:2]{index=2}  

evaluate_model(): Evaluates model performance using R², MAE, and RMSE. :contentReference[oaicite:3]{index=3}  

visualize.py: Contains functions for correlation heatmap and actual vs predicted plots. :contentReference[oaicite:4]{index=4}  

main.py: Runs the full pipeline including data loading, training, evaluation, and visualization. :contentReference[oaicite:5]{index=5}  

---

## 🌐 Streamlit App Features
Sidebar form to enter:

GRE Score  
TOEFL Score  
University Rating  
SOP Strength  
LOR Strength  
CGPA  
Research Experience  

After clicking Predict:

Displays predicted admission probability (%)  
Shows interpretation (Low / Moderate / High likelihood)  
Displays Neural Network loss curve  

Static visualizations:

GRE vs TOEFL (colored by admission probability)  
CGPA distribution  
Pairplot of features  
Actual vs Predicted comparison  

---

## 📁 Project Structure
        .
        ├── data/
        │   ├── raw/                         # Admission.csv dataset
        │   └── processed/                   # Processed dataset
        ├── models/                          # Saved model (NNmodel.pkl, scaler.pkl)
        ├── src/
        │   ├── data/                        # load_and_preprocess_data()
        │   ├── features/                    # build_features()
        │   ├── models/                      # train_NNmodel(), evaluate_model()
        │   ├── visualization/              # plotting functions
        │   └── main.py                     # Main training script
        ├── streamlit.py                    # Streamlit app
        ├── requirements.txt                # Dependencies
        └── README.md                       # Documentation

---

## 🖥 Installation (For Local Deployment)
1. Clone the Repository

git clone https://github.com/yourusername/ucla-admission-predictor.git  
cd ucla-admission-predictor  

2. Install Dependencies  
pip install -r requirements.txt  

3. Train Model & Generate Outputs  
python src/main.py  

This will:

Train the neural network model  
Save it to models/NNmodel.pkl  
Save scaler to models/scaler.pkl  
Generate evaluation outputs and plots  

4. Run the App  
streamlit run streamlit.py  

---

## 📈 Output
models/NNmodel.pkl: Trained Neural Network model  

models/scaler.pkl: Feature scaler  

Visualizations: Plots for analysis and prediction comparison  

---

## 🙌 Thank You!
Thank you for exploring the UCLA Admission Predictor!

This project demonstrates how Neural Networks can be used for regression problems to predict continuous outcomes like admission probability.

Feel free to contribute, raise issues, or share your feedback.
