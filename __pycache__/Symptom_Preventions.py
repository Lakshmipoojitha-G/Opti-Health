import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
X = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]])
y = np.array([0, 1, 1, 0])

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Train the model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X, y)

# Map words to binary values
word_to_binary = {'yes': 1, 'no': 0}

# Get user input
user_input = input("Do you have cough, fever, or headache? (Enter 'yes' or 'no' separated by commas): ")
symptoms = user_input.strip().lower().split(',')

duration_dict = {}
for symptom in symptoms:
    duration = input(f"For {symptom.strip()}, how many days have you been experiencing it? (Enter number of days or 0 if not experiencing): ")
    duration_dict[symptom.strip()] = int(duration)
user_input_array = [word_to_binary[symptom.strip()] for symptom in symptoms]

scaled_input = sc.transform([user_input_array])
prediction = classifier.predict(scaled_input)
if prediction == 1:
    print("Likely symptom detected.")
    for symptom, duration in duration_dict.items():
        if user_input_array[symptoms.index(symptom.strip())] == 1:
            print(f"Preventative measures for {symptom.strip()} (duration: {duration} days):")
            if symptom.strip() == 'cold':
                print("Practicing Good Hygiene, Wash your hands frequently.")
            elif symptom.strip() == 'headache':
                print("Make adjustments to your diet to incorporate appropriate nutrients, Manage stress levels, Rest in a dark and quiet environment")
            elif symptom.strip() == 'fever':
                print("Making sure the room temperature where the person is resting is comfortable, taking a regular bath or a sponge bath using lukewarm water, Take more Fluids")
                if duration > 3:
                    print("Additional measures: If they are children, try to consult a doctor. If they are adults, take rest and try to consume a lot of fluids. If not recovering, consult a doctor.")
else:
    print("No symptom detected.")