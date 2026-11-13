# 🏡 Real Estate Valuation & Price Prediction Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shreyas-Reddy707/Machine-Learning/blob/main/Machine_Learning.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-150458.svg)

## 📌 Project Overview
The **Real Estate Valuation & Price Prediction Engine** is an end-to-end Machine Learning regression project designed to estimate residential property prices in INR (₹) based on physical and locational attributes. 

Developed as a comprehensive Machine Learning demonstration, this repository walks through the complete data science lifecycle: from simulating realistic market dynamics via synthetic data generation, to baseline modeling, hyperparameter tuning, domain-specific feature engineering, and building production-ready `Scikit-Learn` pipelines.

---

## 🎯 Key Objectives & Workflow
1. **Synthetic Data Generation**: Create a realistic dataset of 500 housing records incorporating positive value drivers (e.g., Area, Location Rating), negative depreciation factors (e.g., Age), and controlled random market noise.
2. **Exploratory Data Analysis & Preprocessing**: Inspect data distributions, verify data integrity, and partition the dataset into training ($80\%$) and testing ($20\%$) subsets.
3. **Baseline Modeling & Feature Importance**: Train an initial **Decision Tree Regressor** (`max_depth=7`) and analyze which physical attributes most heavily influence property valuation.
4. **Ensemble Upgrades & Hyperparameter Tuning**: Scale the project using advanced ensemble methods (**Random Forest** and **Gradient Boosting**) and optimize model parameters via 5-fold cross-validated `GridSearchCV`.
5. **Feature Engineering & ML Pipelines**: Construct new domain-specific features and encapsulate data scaling (`StandardScaler`) and regression estimators into clean, leak-proof `Pipeline` architectures.

---

## 📊 Dataset & Feature Specifications

The dataset consists of $500$ synthetic property listings with the following features:

| Feature | Type | Range / Scale | Description |
| :--- | :--- | :--- | :--- |
| **`Area`** | Numeric (`int64`) | $500 - 4000\text{ sq. ft.}$ | Total built-up area of the house. Primary driver of property value. |
| **`Bedrooms`** | Numeric (`int64`) | $1 - 5\text{ units}$ | Total number of bedrooms in the residential unit. |
| **`Bathrooms`** | Numeric (`int64`) | $1 - 4\text{ units}$ | Total number of bathrooms in the residential unit. |
| **`Location_Rating`** | Numeric (`int64`) | $1 - 10\text{ scale}$ | Subjective rating representing neighborhood premium and desirability. |
| **`Age`** | Numeric (`int64`) | $1 - 30\text{ years}$ | Age of the property; acts as a depreciation factor on the overall price. |
| **`Nearby_Amenities`** | Numeric (`int64`) | $1 - 15\text{ count}$ | Number of primary facilities (schools, hospitals, parks, transit) in close proximity. |
| **`Price` *(Target)*** | Numeric (`int64`) | $\text{INR (₹)}$ | Estimated market valuation of the property. |

### Engineered Features (Phase 6)
To capture deeper structural relationships, two additional features were engineered:
* **`Bed_Bath_Ratio`**: $\frac{\text{Bedrooms}}{\text{Bathrooms} + 0.1}$ (Evaluates room balance while preventing division by zero).
* **`Area_per_Room`**: $\frac{\text{Area}}{\text{Bedrooms} + \text{Bathrooms} + 1}$ (Measures the spaciousness of individual rooms).

---

## 🧠 Model Performance & Evaluation

We evaluated three tree-based regression algorithms across standard error metrics: **Mean Absolute Error ($MAE$)**, **Root Mean Squared Error ($RMSE$)**, and **Coefficient of Determination ($R^2\text{ Score}$)**.

### 1. Standard & Tuned Models Comparison
Using `GridSearchCV`, the Random Forest model was tuned across `n_estimators`, `max_depth`, and `min_samples_split`, identifying optimal parameters at `{'max_depth': 15, 'min_samples_split': 2, 'n_estimators': 200}`.

| Model | $MAE$ (₹) | $RMSE$ (₹) | $R^2\text{ Score}$ |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting (Base)** | **₹320,323.24** | **₹423,031.89** | **0.9858** |
| **Random Forest (Tuned)** | ₹420,673.07 | ₹540,194.71 | 0.9768 |
| **Decision Tree (Baseline)** | ₹626,687.45 | ₹799,135.52 | 0.9492 |

### 2. End-to-End Pipeline Evaluation (with Engineered Features)
Wrapping `StandardScaler` and the regressors into a `Scikit-Learn Pipeline` evaluated across 5-Fold Cross-Validation yielded highly stable results:

| Pipeline Model | 5-Fold CV Mean $R^2$ | CV Std $R^2$ | Test $R^2\text{ Score}$ | Test $MAE$ (₹) |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting (Pipeline)** | **0.9873** | **$\pm 0.0008$** | **0.9855** | **₹322,953.51** |
| **Random Forest (Pipeline)** | 0.9783 | $\pm 0.0019$ | 0.9754 | ₹435,969.16 |
| **Decision Tree (Pipeline)** | 0.9550 | $\pm 0.0032$ | 0.9448 | ₹675,380.56 |

> **Key Takeaway:** **Gradient Boosting** consistently outperformed both the baseline Decision Tree and the hyperparameter-tuned Random Forest, explicar over **$98.5\%$** of the variance in housing prices and reducing the Mean Absolute Error by nearly $50\%$ compared to the baseline.

---

## 📈 Visualizations Included in the Notebook
* **Feature Importance Bar Chart**: Highlights that `Area` is the dominant predictor of housing price, followed by `Location_Rating` and `Age`.
* **Model $R^2$ Score Comparison**: A horizontal bar plot visually contrasting the accuracy improvements achieved through ensemble methods.
* **Actual vs. Predicted Scatter Plot**: Demonstrates the tight alignment of the Gradient Boosting predictions along the ideal $45^\circ$ identity line.

---

## 🛠️ Tech Stack & Libraries
* **Programming Language**: Python 3
* **Data Manipulation & Math**: `pandas`, `numpy`
* **Data Visualization**: `matplotlib`, `seaborn`
* **Machine Learning**: `scikit-learn` (`DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `GridSearchCV`, `Pipeline`, `StandardScaler`, `cross_val_score`)

---

## 🚀 How to Run the Project

### Option 1: Run in Google Colab (Recommended)
Click the **Open in Colab** badge at the top of this README to launch the notebook directly in your browser with zero local setup required.

### Option 2: Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shreyas-Reddy707/Machine-Learning.git
   cd Machine-Learning
   ```

2. **Install the required dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Machine_Learning.ipynb
   ```

---

## 🔮 Sample Out-of-Sample Inference
The notebook includes an interactive inference cell where you can pass custom property parameters to predict its market value:

```python
new_house = pd.DataFrame({
    'Area': [2400],
    'Bedrooms': [3],
    'Bathrooms': [2],
    'Location_Rating': [8],
    'Age': [5],
    'Nearby_Amenities': [7]
})

predicted_price = model.predict(new_house)
print("Predicted House Price: ₹", int(predicted_price[0]))
# Output: Predicted House Price: ₹ 11080074
```