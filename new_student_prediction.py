import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("student_placement.csv")

# Encode Internship
internship_encoder = LabelEncoder()
df["Internship"] = internship_encoder.fit_transform(df["Internship"])

# Encode Placement Status
placement_encoder = LabelEncoder()
df["Placement_Status"] = placement_encoder.fit_transform(df["Placement_Status"])

# Separate input and output
X = df.drop(["Student_ID", "Placement_Status"], axis=1)
y = df["Placement_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Normalize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# --------------------------------
# NEW STUDENT DETAILS
# --------------------------------

print("\nEnter New Student Details")

cgpa = float(input("CGPA: "))
ssc = float(input("SSC Percentage: "))
diploma_hsc = float(input("Diploma/HSC Percentage: "))
aptitude = int(input("Aptitude Score: "))
communication = int(input("Communication Score (1-10): "))
technical = int(input("Technical Skill Score (1-10): "))
internship = input("Internship (Yes/No): ")
projects = int(input("Number of Projects: "))
certifications = int(input("Number of Certifications: "))
attendance = float(input("Attendance Percentage: "))

# Convert Internship
if internship.lower() == "yes":
    internship_value = internship_encoder.transform(["Yes"])[0]
else:
    internship_value = internship_encoder.transform(["No"])[0]

# Create new student data
new_student = pd.DataFrame({
    "CGPA": [cgpa],
    "SSC_Percentage": [ssc],
    "Diploma_or_HSC_Percentage": [diploma_hsc],
    "Aptitude_Score": [aptitude],
    "Communication_Score": [communication],
    "Technical_Skill_Score": [technical],
    "Internship": [internship_value],
    "Projects": [projects],
    "Certifications": [certifications],
    "Attendance_Percentage": [attendance]
})

# Normalize
new_student_scaled = scaler.transform(new_student)

# Prediction
prediction = model.predict(new_student_scaled)

# Result
result = placement_encoder.inverse_transform(prediction)

print("\n-----------------------------")
print("Placement Prediction:", result[0])
print("-----------------------------")