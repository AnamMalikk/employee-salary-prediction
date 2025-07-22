# 💼 Employee Salary Prediction 


## 🎯 Objective
The primary objective of this project is to analyze demographic and work-related attributes to **predict whether an individual earns more than $50,000 annually**. Using the Adult Census Income dataset, this project demonstrates the complete machine learning workflow — from data cleaning to model deployment.


## 📊 Dataset Overview
The dataset contains information on over 48,000 individuals, including features like:
- Age
- Education Level
- Occupation
- Hours per week
- Workclass
- Marital Status
- Native Country
- Gender  
...and more.

🔗 **Source**:  
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)  
- [Kaggle - Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)


## 🔍 Methodology
Our pipeline is organized as follows:
1. **Data Loading & Inspection**
   - Loaded dataset into Pandas DataFrame
   - Identified data types, null values, and distribution

2. **Exploratory Data Analysis**
   - Used Matplotlib and Seaborn for insightful visualizations
   - Boxplots for outliers, count plots for categories

3. **Data Cleaning & Transformation**
   - Removed outliers from `age` and `educational-num`
   - Replaced missing values marked as `'?'` with `'Other'`
   - Dropped less relevant columns like `education` (in favor of `educational-num`)

4. **Encoding & Scaling**
   - Applied Label Encoding for categorical variables
   - Used Min-Max Scaling for numerical features

5. **Class Imbalance Handling**
   - Implemented SMOTE (Synthetic Minority Over-sampling) to balance target classes

6. **Model Building**
   - Trained and evaluated multiple models:
     - Logistic Regression
     - Random Forest
     - SVM
     - KNN
     - Gradient Boosting
     - MLP Classifier 

7. **Deployment**
   - Built a Streamlit web app for:
     - Real-time single prediction
     - CSV batch predictions
     - Clean and styled UI with gradient effects


## 🧠 Project Highlights
- ✔️ Complete ML pipeline from raw data to web app
- 📈 Focus on model accuracy and class balance
- 💡 Emphasis on readability, maintainability, and visualization
- 🌐 Deployed via **Streamlit** for easy accessibility


## 📌 Folder Structure

```

employee-salary-prediction/
│
├── Employee_Salary_Prediction.ipynb       # Main Colab notebook
├── app.py                                 # Streamlit app for deployment
├── best_model.pkl                         # Serialized ML model
├── requirements.txt                       # Project dependencies
└── README.md                              # This file

````


## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Adult Census Income dataset (adult.csv)

### Installation
```bash
git clone https://github.com/AnamMalikk/employee-salary-prediction.git
cd employee-salary-prediction
pip install -r requirements.txt
```

### Dataset Setup
1. Download the Adult Census Income dataset.
2. Place the `adult.csv` file in the project root directory
3. Update the file path in `app.py` if needed

### Run the Application
```bash
streamlit run app.py
```



## 📝 Sample Input Format

### Single Prediction Input:
When using the Streamlit app, provide input in this format:

| Feature | Example Value | Description |
|---------|---------------|-------------|
| Age | 39 | Age in years |
| Workclass | Private | Type of employment |
| Education-num | 13 | Years of education |
| Marital-status | Never-married | Marital status |
| Occupation | Adm-clerical | Job type |
| Relationship | Not-in-family | Family relationship |
| Race | White | Race |
| Gender | Male | Gender |
| Capital-gain | 2174 | Capital gains |
| Capital-loss | 0 | Capital losses |
| Hours-per-week | 40 | Working hours per week |
| Native-country | United-States | Country of origin |

### CSV Batch Prediction Format:
For batch predictions, upload a CSV file with these columns:
```csv
age,workclass,education-num,marital-status,occupation,relationship,race,gender,capital-gain,capital-loss,hours-per-week,native-country
39,Private,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States
50,Self-emp-not-inc,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States
38,Private,9,Divorced,Handlers-cleaners,Not-in-family,White,Male,0,0,40,United-States
```

**Prediction Output:** 
- `<=50K` - Earns $50,000 or less annually
- `>50K` - Earns more than $50,000 annually


## �️ Troubleshooting
### Common Issues:
1. **"adult.csv not found" error:**
   - Ensure the dataset file is in the project root directory
   - Check the file path in `app.py` line 13

2. **Module import errors:**
   - Run `pip install -r requirements.txt`
   - Ensure you're using Python 3.7+

3. **Streamlit won't start:**
   - Try `python -m streamlit run app.py`
   - Check if port 8501 is available


## 🔮 Future Enhancements
* **Model Explainability**: Add SHAP/LIME to explain why the model makes specific predictions
* **Cloud Deployment**: Deploy to platforms like **AWS**, **Heroku**, or **Streamlit Cloud** for public access
* **User Authentication**: Include login system for private and secure predictions
* **Analytics Dashboard**: Add usage statistics and prediction logging features
* **Model Monitoring**: Track model performance and data drift over time
* **API Integration**: Create REST API endpoints for external applications


## 🤝 Contributing
Contributions are welcome! If you’d like to improve something or report an issue, feel free to submit a pull request or open an issue.


## 📬 Contact
**Anam** – MCA Student, IGDTUW
📧 Email: [mkanam.del@gmail.com](mailto:mkanam.del@gmail.com)


⭐️ If you found this project useful, please give it a **star** on GitHub!


