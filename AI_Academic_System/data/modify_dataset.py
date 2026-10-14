import pandas as pd

df = pd.read_csv("student_performance_large_dataset.csv")

stress_map = {"Low": 1, "Medium": 2, "High": 3}

df["Stress_Num"] = df["Self_Reported_Stress_Level"].map(stress_map)

df["Burnout_Score"] = (
    df["Stress_Num"] * 2 +
    (10 - df["Sleep_Hours_per_Night"]) * 0.5 +
    df["Time_Spent_on_Social_Media (hours/week)"] * 0.2 +
    df["Study_Hours_per_Week"] * 0.1
)

df["Burnout_Level"] = pd.cut(
    df["Burnout_Score"],
    bins=[-100, 5, 8, 100],
    labels=["Low", "Medium", "High"]
)

df["Productivity_Score"] = (
    df["Assignment_Completion_Rate (%)"] * 0.3 +
    df["Exam_Score (%)"] * 0.4 +
    df["Attendance_Rate (%)"] * 0.2 +
    df["Sleep_Hours_per_Night"] * 2
)

df["Mental_Health_Risk"] = pd.cut(
    df["Stress_Num"] + (10 - df["Sleep_Hours_per_Night"]),
    bins=[-100, 5, 8, 100],
    labels=["Low", "Medium", "High"]
)

df.to_csv("student_burnout_modified.csv", index=False)

print("✅ Dataset Modified Successfully")
