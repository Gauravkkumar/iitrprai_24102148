# 🎓 Student Exam Score Prediction System

## 🎯 Project Overview
This project demonstrates practical **machine learning deployment for educational analytics**.  
It predicts **student exam scores** based on multiple features such as study hours, previous scores, attendance, and extracurricular activities.  
The system provides **real-time predictions, model training**, and **comprehensive analytics** through a FastAPI backend and Streamlit UI.

**Student:** Gaurav Kumar
---

## ✨ Features
- 🤖 **Multiple ML Models:** Compare Linear Regression vs Decision Tree  
- 📊 **Interactive UI:** Streamlit-based web interface  
- 🚀 **Real-time Predictions:** Instant score predictions  
- 🎛️ **Parameter Tuning:** Adjust model parameters in real-time  
- 📈 **Data Visualization:** Correlation heatmaps, distributions, and analytics  
- 🔧 **Model Training:** Train models with custom parameters  
- 📱 **Responsive Design:** User-friendly educational interface  

---

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.10 or higher  
- pip (Python package manager)


### **Step 1: Create Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

---


### **Method : Manual Startup**

#### Terminal 1: Start Backend Server
```bash
python app.py
```
Backend will run at: **http://localhost:8000**

#### Terminal 2: Start Frontend UI
```bash
streamlit run streamlit_app.py
```
Frontend will run at: **http://localhost:8501**

---

## 📁 Project Structure
```text
student-score-prediction/
│
├── main.py                 # FastAPI backend server
├── frontend.py       # Streamlit frontend application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation

```

## 🎮 Usage Guide

### **1. Data Explorer**
- View sample student dataset  
- Analyze feature correlations  
- Explore data distributions  
- Generate new synthetic data  

### **2. Model Training**
- Adjust training parameters  
- Train Linear Regression and Decision Tree models  
- Compare model performance metrics  
- Tune hyperparameters in real-time  

### **3. Prediction Lab**
- Input student parameters using sliders  
- Get instant predictions from both models  
- Compare predictions visually  
- Test different student profiles  

### **4. Performance Analytics**
- View R² and MAE metrics  
- Compare model performance  
- Analyze feature importance  
- Get model recommendations  

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:-------|:----------|:-------------|
| POST | `/train` | Train ML models with parameters |
| POST | `/predict` | Predict exam score for a student |
| GET  | `/data` | Get sample dataset |
| GET  | `/metrics` | Get model performance metrics |
| POST | `/generate_new_data` | Generate new synthetic dataset |

### Example Usage
```python
import requests

# Train models
training_config = {
    "test_size": 0.2,
    "random_state": 42,
    "max_depth": 5,
    "min_samples_split": 2
}
requests.post("http://localhost:8000/train", json=training_config)

# Make prediction
student_data = {
    "hours_studied": 6.0,
    "previous_scores": 85.0,
    "attendance": 90.0,
    "extracurricular": 2.0,
    "parental_education": 3
}
requests.post("http://localhost:8000/predict", json=student_data)
```

---

## 📊 Dataset Features

| Feature | Description | Range |
|:--|:--|:--|
| hours_studied | Daily study hours | 1–10 hours |
| previous_scores | Historical exam scores | 40–100 |
| attendance | Class attendance percentage | 60–100% |
| extracurricular | Hours in extracurricular activities | 0–5 hours |
| parental_education | Education level of parents | 1–5 scale |
| final_score | Target variable: Exam score | 0–100 |

---

## 🤖 Machine Learning Models

### **Linear Regression**
- Baseline model for linear relationships  
- Simple, fast, and interpretable  

### **Decision Tree Regression**
- Captures non-linear patterns  
- Handles complex relationships  
- Configurable depth and splits  

---

## 📈 Performance Metrics

| Model | R² Score | Mean Absolute Error |
|:--|:--:|:--:|
| Linear Regression | 0.72 | 4.2 |
| Decision Tree | 0.85 | 3.2 |


---

## 🚀 Deployment

### **Local Deployment**
Follow the installation steps above to run the app locally.

### **Cloud Deployment (Future)**
- Deploy FastAPI backend on **Heroku** or **Railway**  
- Deploy Streamlit frontend on **Streamlit Cloud**  
- Use cloud database for persistent storage  

---

## 🛠️ Development

### **Adding New Features**
1. Fork the repository  
2. Create a feature branch  
3. Implement your changes  
4. Test thoroughly  
5. Submit a pull request  

### **Code Style**
- Follow **PEP 8** guidelines  
- Use meaningful variable names  
- Add comments for complex logic  
- Include docstrings for functions  

---

## 🤝 Contributing
Contributions are welcome!  
Feel free to open **issues** or **pull requests** for:
- 🐞 Bug fixes  
- 🌟 New features  
- 📝 Documentation improvements  
- ⚙️ Performance enhancements
---

## 📝 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author
**Gaurav Kumar**
---

## 🙏 Acknowledgments
- FastAPI and Streamlit communities for excellent documentation  
- Scikit-learn team for robust ML libraries  
- Educational data mining research that inspired this project  


