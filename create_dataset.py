import pandas as pd
import numpy as np

np.random.seed(42)

n = 250

data = {
    "Student_ID": [f"S{i:03d}" for i in range(1, n + 1)],
    "CGPA": np.round(np.random.uniform(5.5, 9.7, n), 2),
    "SSC_Percentage": np.round(np.random.uniform(55, 95, n), 2),
    "Diploma_or_HSC_Percentage": np.round(np.random.uniform(55, 95, n), 2),
    "Aptitude_Score": np.random.randint(35, 96, n),
    "Communication_Score": np.random.randint(4, 11, n),
    "Technical_Skill_Score": np.random.randint(4, 11, n),
    "Internship": np.random.choice(["Yes", "No"], n),
    "Projects": np.random.randint(0, 5, n),
    "Certifications": np.random.randint(0, 6, n),
    "Attendance_Percentage": np.round(np.random.uniform(60, 99, n), 2)
}

df = pd.DataFrame(data)

score = (
    df["CGPA"] * 8
    + df["SSC_Percentage"] * 0.20
    + df["Diploma_or_HSC_Percentage"] * 0.15
    + df["Aptitude_Score"] * 0.35
    + df["Communication_Score"] * 2
    + df["Technical_Skill_Score"] * 2
    + (df["Internship"] == "Yes") * 8
    + df["Projects"] * 3
    + df["Certifications"] * 1.5
    + df["Attendance_Percentage"] * 0.08
)

threshold = score.median()

df["Placement_Status"] = np.where(
    score >= threshold,
    "Placed",
    "Not Placed"
)

df.to_csv("student_placement.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print("\nFirst 10 records:")
print(df.head(10))