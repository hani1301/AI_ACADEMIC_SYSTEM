"""
generate_charts.py

Run AFTER train_multitask_model.py

Required files:
results/history.pkl
results/artifacts.pkl
models/multitask_model.h5
models/encoders.pkl
"""

import os
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# Create charts folder
# =====================================================

os.makedirs("charts", exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 300
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"charts/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved charts/{name}.png")


# =====================================================
# Load Saved Files
# =====================================================

with open("results/artifacts.pkl", "rb") as f:
    art = pickle.load(f)

with open("results/history.pkl", "rb") as f:
    hist = pickle.load(f)

# IMPORTANT FIX FOR KERAS 3
model = load_model(
    "models/multitask_model.h5",
    compile=False
)

encoders = joblib.load("models/encoders.pkl")

burnout_labels = art["burnout_classes"]
mental_labels = art["mental_classes"]

X_test_s = art["X_test_s"]

yb_test_i = art["yb_test_i"]
ym_test_i = art["ym_test_i"]

yp_test = art["yp_test"]


# =====================================================
# Predict
# =====================================================

pred_burn, pred_ment, pred_prod = model.predict(
    X_test_s,
    verbose=0
)

yb_pred = np.argmax(pred_burn, axis=1)
ym_pred = np.argmax(pred_ment, axis=1)
yp_pred = pred_prod.flatten()


# =====================================================
# Training Convergence
# =====================================================

epochs = range(1, len(hist["loss"]) + 1)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    epochs,
    hist["loss"],
    linewidth=2,
    label="Training Loss"
)

ax.plot(
    epochs,
    hist["val_loss"],
    linewidth=2,
    linestyle="--",
    label="Validation Loss"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Convergence")
ax.legend()

save(fig, "training_convergence")


# =====================================================
# Confusion Matrix Function
# =====================================================

def plot_confusion(
    y_true,
    y_pred,
    labels,
    title,
    filename
):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black"
            )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    save(fig, filename)


plot_confusion(
    yb_test_i,
    yb_pred,
    burnout_labels,
    "Burnout Confusion Matrix",
    "confusion_burnout"
)

plot_confusion(
    ym_test_i,
    ym_pred,
    mental_labels,
    "Mental Health Confusion Matrix",
    "confusion_mentalhealth"
)

# =====================================================
# Actual vs Predicted Productivity
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(
    yp_test,
    yp_pred,
    alpha=0.7
)

ax.plot(
    [yp_test.min(), yp_test.max()],
    [yp_test.min(), yp_test.max()],
    'r--',
    linewidth=2
)

ax.set_xlabel("Actual Productivity Score")
ax.set_ylabel("Predicted Productivity Score")
ax.set_title("Actual vs Predicted Productivity")

save(fig,"actual_vs_predicted_productivity")


# =====================================================
# Residual Plot
# =====================================================

residuals = yp_test - yp_pred

fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(
    yp_pred,
    residuals,
    alpha=0.7
)

ax.axhline(
    y=0,
    color="red",
    linestyle="--"
)

ax.set_xlabel("Predicted Productivity")
ax.set_ylabel("Residual Error")
ax.set_title("Residual Plot")

save(fig,"residual_plot")


# =====================================================
# Error Distribution
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.hist(
    residuals,
    bins=20
)

ax.set_xlabel("Prediction Error")
ax.set_ylabel("Frequency")
ax.set_title("Prediction Error Distribution")

save(fig,"error_distribution")


# =====================================================
# Productivity Score Distribution
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.hist(
    yp_test,
    bins=20
)

ax.set_xlabel("Productivity Score")
ax.set_ylabel("Frequency")
ax.set_title("Productivity Score Distribution")

save(fig,"productivity_distribution")


# =====================================================
# Actual vs Predicted Line Plot
# =====================================================

n = min(100, len(yp_test))

fig, ax = plt.subplots(figsize=(9,5))

ax.plot(
    range(n),
    yp_test[:n],
    linewidth=2,
    label="Actual"
)

ax.plot(
    range(n),
    yp_pred[:n],
    linewidth=2,
    linestyle="--",
    label="Predicted"
)

ax.set_xlabel("Test Samples")
ax.set_ylabel("Productivity Score")
ax.set_title("Actual vs Predicted Productivity (First 100 Samples)")
ax.legend()

save(fig,"actual_vs_predicted_line")


# =====================================================
# Regression Fit
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.scatter(
    yp_test,
    yp_pred,
    alpha=0.6
)

m, b = np.polyfit(yp_test, yp_pred, 1)

idx = np.argsort(yp_test)

ax.plot(
    yp_test[idx],
    m * yp_test[idx] + b,
    linewidth=2
)

ax.set_xlabel("Actual Productivity")
ax.set_ylabel("Predicted Productivity")
ax.set_title("Regression Fit")

save(fig,"regression_fit")

# =====================================================
# Burnout Class Distribution
# =====================================================

unique, counts = np.unique(yb_test_i, return_counts=True)

fig, ax = plt.subplots(figsize=(6,5))

ax.bar(
    burnout_labels,
    counts
)

ax.set_xlabel("Burnout Level")
ax.set_ylabel("Number of Students")
ax.set_title("Burnout Class Distribution")

save(fig,"burnout_distribution")


# =====================================================
# Mental Health Class Distribution
# =====================================================

unique, counts = np.unique(ym_test_i, return_counts=True)

fig, ax = plt.subplots(figsize=(6,5))

ax.bar(
    mental_labels,
    counts
)

ax.set_xlabel("Mental Health Risk")
ax.set_ylabel("Number of Students")
ax.set_title("Mental Health Risk Distribution")

save(fig,"mental_health_distribution")


# =====================================================
# Evaluation Metrics
# =====================================================

burn_acc = accuracy_score(yb_test_i, yb_pred) * 100
burn_pre = precision_score(
    yb_test_i,
    yb_pred,
    average="weighted",
    zero_division=0
) * 100

burn_rec = recall_score(
    yb_test_i,
    yb_pred,
    average="weighted",
    zero_division=0
) * 100

burn_f1 = f1_score(
    yb_test_i,
    yb_pred,
    average="weighted",
    zero_division=0
) * 100


mental_acc = accuracy_score(ym_test_i, ym_pred) * 100

mental_pre = precision_score(
    ym_test_i,
    ym_pred,
    average="weighted",
    zero_division=0
) * 100

mental_rec = recall_score(
    ym_test_i,
    ym_pred,
    average="weighted",
    zero_division=0
) * 100

mental_f1 = f1_score(
    ym_test_i,
    ym_pred,
    average="weighted",
    zero_division=0
) * 100


# =====================================================
# Accuracy Comparison
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.bar(
    ["Burnout", "Mental Health"],
    [burn_acc, mental_acc]
)

ax.set_ylim(0,100)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy Comparison")

save(fig,"accuracy_comparison")


# =====================================================
# Precision Comparison
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.bar(
    ["Burnout", "Mental Health"],
    [burn_pre, mental_pre]
)

ax.set_ylim(0,100)

ax.set_ylabel("Precision (%)")
ax.set_title("Precision Comparison")

save(fig,"precision_comparison")


# =====================================================
# Recall Comparison
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.bar(
    ["Burnout", "Mental Health"],
    [burn_rec, mental_rec]
)

ax.set_ylim(0,100)

ax.set_ylabel("Recall (%)")
ax.set_title("Recall Comparison")

save(fig,"recall_comparison")


# =====================================================
# F1 Score Comparison
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

ax.bar(
    ["Burnout", "Mental Health"],
    [burn_f1, mental_f1]
)

ax.set_ylim(0,100)

ax.set_ylabel("F1 Score (%)")
ax.set_title("F1 Score Comparison")

save(fig,"f1score_comparison")

# =====================================================
# Regression Metrics
# =====================================================

mae = mean_absolute_error(yp_test, yp_pred)
mse = mean_squared_error(yp_test, yp_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yp_test, yp_pred)


# =====================================================
# Combined Metrics Comparison
# =====================================================

fig, ax = plt.subplots(figsize=(8,5))

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

burn_values = [burn_acc, burn_pre, burn_rec, burn_f1]
mental_values = [mental_acc, mental_pre, mental_rec, mental_f1]

x = np.arange(len(metrics))
width = 0.35

ax.bar(
    x - width/2,
    burn_values,
    width,
    label="Burnout"
)

ax.bar(
    x + width/2,
    mental_values,
    width,
    label="Mental Health"
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.set_ylim(0, 100)

ax.set_ylabel("Percentage (%)")
ax.set_title("Classification Performance Comparison")
ax.legend()

save(fig, "classification_metrics_comparison")


# =====================================================
# Productivity Regression Metrics
# =====================================================

fig, ax = plt.subplots(figsize=(6,5))

metric_names = ["MAE", "RMSE"]
metric_values = [mae, rmse]

ax.bar(metric_names, metric_values)

ax.set_ylabel("Error")
ax.set_title("Productivity Regression Error")

save(fig, "productivity_error_metrics")


# =====================================================
# Print Final Results
# =====================================================

print("\n" + "="*65)
print("PROPOSED MULTITASK MODEL - FINAL RESULTS")
print("="*65)

print("\nBurnout Classification")
print("-"*35)
print(f"Accuracy  : {burn_acc:.2f}%")
print(f"Precision : {burn_pre:.2f}%")
print(f"Recall    : {burn_rec:.2f}%")
print(f"F1 Score  : {burn_f1:.2f}%")

print("\nMental Health Classification")
print("-"*35)
print(f"Accuracy  : {mental_acc:.2f}%")
print(f"Precision : {mental_pre:.2f}%")
print(f"Recall    : {mental_rec:.2f}%")
print(f"F1 Score  : {mental_f1:.2f}%")

print("\nProductivity Regression")
print("-"*35)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

print("\n" + "="*65)
print("Charts generated successfully.")
print("All figures are saved in the 'charts' folder.")
print("="*65)