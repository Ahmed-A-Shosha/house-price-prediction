# 🏠 House Price Prediction — Machine Learning Project

> Predicting house prices using multiple ML models with 98.9% accuracy.

**By: Ahmed Adel Shosha** | AI Engineer & ML Specialist

---

## 📊 Results

![House Price Prediction Results](house_price_results.png)

---

## 🎯 Project Overview

A complete Machine Learning pipeline that predicts house prices based on 11 features including area, number of rooms, age, location, and amenities.

The project covers the full ML workflow:
- Data generation & exploration (EDA)
- Feature engineering
- Training & comparing 3 models
- Evaluating performance
- Predicting prices for new houses

---

## 🏆 Model Performance

| Model | R² Score | MAE |
|-------|----------|-----|
| **Linear Regression** | **98.9%** ✅ | $12,604 |
| Gradient Boosting | 97.8% | $17,415 |
| Random Forest | 95.5% | $25,841 |

---

## 🔧 Features Used

| Feature | Description |
|---------|-------------|
| `area_sqft` | Total area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Age of the house |
| `distance_center` | Distance from city center (km) |
| `has_garage` | Garage availability (0/1) |
| `has_pool` | Pool availability (0/1) |
| `floor_number` | Floor number |
| `neighborhood` | Premium / Standard / Budget |
| `rooms_total` | Total rooms (engineered) |
| `is_new` | House under 5 years old (engineered) |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `house_price.py`
3. Run all cells

### Option 2: Local
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the project
python house_price.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## 📁 Project Structure

```
house-price-prediction/
│
├── house_price.py           # Main ML code
├── house_price_results.png  # Visualizations
└── README.md                # Project documentation
```

---

## 🧠 What I Learned

- Data preprocessing & feature engineering
- Training and comparing multiple ML models
- Model evaluation metrics (R², MAE, RMSE)
- Data visualization with Matplotlib & Seaborn

---

## 👤 Author

**Ahmed Adel Shosha**
- 🌐 Portfolio: [ahmed-a-shosha.github.io](https://ahmed-a-shosha.github.io)
- 💼 LinkedIn: [linkedin.com/in/ahmedadelshosha](https://linkedin.com/in/ahmedadelshosha)
- 🐙 GitHub: [github.com/Ahmed-A-Shosha](https://github.com/Ahmed-A-Shosha)
