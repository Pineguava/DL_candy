# Candy Popularity Prediction with TensorFlow & Keras

A deep learning project using **TensorFlow / Keras** to predict candy popularity (`winpercent`) from product attributes in the **`candy.csv`** dataset.

---

## Dataset Overview

The dataset (`candy.csv`) contains attributes for ~85 candies.

### Features & Target

| Column | Type | Description |
| :--- | :--- | :--- |
| `competitorname` | `string` | Name of the candy (dropped before training) |
| `chocolate` | `int` (`0`/`1`) | Does it contain chocolate? |
| `fruity` | `int` (`0`/`1`) | Is it fruit-flavored? |
| `caramel` | `int` (`0`/`1`) | Is there caramel in the candy? |
| `peanutyalmondy` | `int` (`0`/`1`) | Does it contain peanuts, peanut butter, or almonds? |
| `nougat` | `int` (`0`/`1`) | Does it contain nougat? |
| `crispedricewafer` | `int` (`0`/`1`) | Does it contain crisped rice, wafers, or a cookie component? |
| `hard` | `int` (`0`/`1`) | Is it a hard candy? |
| `bar` | `int` (`0`/`1`) | Is it a candy bar? |
| `pluribus` | `int` (`0`/`1`) | Is it one of many candies in a bag or box? |
| `sugarpercent` | `float` (`0`–`1`) | Percentile of sugar content within the dataset |
| `pricepercent` | `float` (`0`–`1`) | Unit price percentile compared to the rest of the set |
| **`winpercent`** | `float` | **Target:** Overall win percentage from head-to-head survey matchups |

*(Note: Some versions of the CSV also include an `id` index column, which is automatically dropped if present.)*

---

## Model & Approach

Because `candy.csv` is a small tabular dataset, the neural network is intentionally compact and regularized to avoid overfitting:

1. **Preprocessing:**
   - Drops non-feature columns (`competitorname`, and `id` if present).
   - Splits data 80/20 into training and validation sets (`train_test_split`).
   - Standardizes input features with `StandardScaler` (fit on the training set only).
2. **Architecture (`keras.Sequential`):**
   - `Dense(64, activation="relu")` → `BatchNormalization` → `Dropout(0.3)`
   - `Dense(32, activation="relu")` → `BatchNormalization` → `Dropout(0.2)`
   - `Dense(1)` (linear output for regression)
3. **Training:**
   - Optimizer: `adam`
   - Loss & Metric: Mean Absolute Error (`mae`)
   - Callback: `EarlyStopping` monitoring `val_loss` (`patience=20`, `restore_best_weights=True`)

---
