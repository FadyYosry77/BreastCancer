# Project Documentation: Breast Cancer Prediction
### Project Aim
* **Primary Goal**: To automate the detection of breast cancer by classifying tumors as Malignant or Benign based on diagnostic measurements.
* **Comparative Analysis**: To evaluate and compare the predictive performance of Deep Learning (Neural Networks) versus classical Machine Learning (Support Vector Machines).
## 1. Data Loading & Exploration
- Load dataset  
- Preview data  
- Check missing values  
- Data summary  

## 2. Data Preprocessing
- Remove duplicates  
- Encode target variable  
- Handle outliers (IQR method)  

## 3. Exploratory Data Analysis (EDA)
- Correlation heatmap  
- Feature–target correlation  
- Pairplot of key features  
- Feature distribution (histograms)  

## 4. Feature Engineering & Scaling
- Separate features and target  
- Standardize features  

## 5. Dimensionality Reduction (PCA)
- Apply PCA (95% variance retention)  

## 6. Data Splitting
- Train–test split (80/20)  

## 7. Neural Network Model (Model 1)
- Build architecture  
- Compile model  
- Train model  
- Evaluate performance  
- Plot learning curves  
- Confusion matrix  

## 8. Neural Network Model (Model 2)
- Retrain model  
- Performance evaluation  
- Confusion matrix  

## 9. Support Vector Machine (SVM)
- Train–test split (scaled data)  
- Hyperparameter tuning (kernel, C)  
- Model evaluation  
- Confusion matrix  

## 10. Model Comparison
- ROC curve comparison  
- Performance metrics comparison (Accuracy, Precision, Recall, F1-score)  
