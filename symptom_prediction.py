# Importing libraries
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
def predict(): 
	entered_time = datetime.now()
	# print("Entered Predict",entered_time)
	# Reading the train.csv by removing the 
	# last column since it's an empty column
	DATA_PATH = "dataset/Training.csv"
	data = pd.read_csv(DATA_PATH).dropna(axis = 1)

	# Checking whether the dataset is balanced or not
	disease_counts = data["prognosis"].value_counts()
	temp_df = pd.DataFrame({
		"Disease": disease_counts.index,
		"Counts": disease_counts.values
	})

	# plt.figure(figsize = (18,8))
	sns.barplot(x = "Disease", y = "Counts", data = temp_df)
	# plt.xticks(rotation=90)
	##plt.show()
	# Encoding the target value into numerical
	# value using LabelEncoder
	encoder = LabelEncoder()
	data["prognosis"] = encoder.fit_transform(data["prognosis"])

	X = data.iloc[:,:-1]
	y = data.iloc[:, -1]
	X_train, X_test, y_train, y_test =train_test_split(
	X, y, test_size = 0.2, random_state = 24)

	#print(f"Train#: {X_train.shape}, {y_train.shape}")
	#print(f"Test#: {X_test.shape}, {y_test.shape}")


	# Defining scoring metric for k-fold cross validation
	def cv_scoring(estimator, X, y):
		return accuracy_score(y, estimator.predict(X))

	# Initializing Models
	models = {
		"SVC":SVC(),
		"Gaussian NB":GaussianNB(),
		"Random Forest":RandomForestClassifier(random_state=18)
	}

	# Producing cross validation score for the models
	for model_name in models:
		model = models[model_name]
		scores = cross_val_score(model, X, y, cv = 10, 
								n_jobs = -1, 
								scoring = cv_scoring)
		#print("=="*30)
		#print(model_name)
		#print(f"Scores#: {scores}")
		#print(f"Mean Score#: {np.mean(scores)}")

	# Training and testing SVM Classifier
	svm_model = SVC()
	svm_model.fit(X_train, y_train)
	preds = svm_model.predict(X_test)

	#print(f"Accuracy on train data by SVM Classifier\
	#: {accuracy_score(y_train, svm_model.predict(X_train))*100}")

	#print(f"Accuracy on test data by SVM Classifier\
	#: {accuracy_score(y_test, preds)*100}")
	cf_matrix = confusion_matrix(y_test, preds)
	# plt.figure(figsize=(12,8))
	sns.heatmap(cf_matrix, annot=True)

	# Training and testing Naive Bayes Classifier
	nb_model = GaussianNB()
	nb_model.fit(X_train, y_train)
	preds = nb_model.predict(X_test)
	#print(f"Accuracy on train data by Naive Bayes Classifier\
	#: {accuracy_score(y_train, nb_model.predict(X_train))*100}")

	#print(f"Accuracy on test data by Naive Bayes Classifier\
	#: {accuracy_score(y_test, preds)*100}")
	cf_matrix = confusion_matrix(y_test, preds)

	# Training and testing Random Forest Classifier
	rf_model = RandomForestClassifier(random_state=18)
	rf_model.fit(X_train, y_train)
	preds = rf_model.predict(X_test)
	#print(f"Accuracy on train data by Random Forest Classifier\
	#: {accuracy_score(y_train, rf_model.predict(X_train))*100}")

	#print(f"Accuracy on test data by Random Forest Classifier\
	#: {accuracy_score(y_test, preds)*100}")

	cf_matrix = confusion_matrix(y_test, preds)

	# Training the models on whole data
	final_svm_model = SVC()
	final_nb_model = GaussianNB()
	final_rf_model = RandomForestClassifier(random_state=18)
	final_svm_model.fit(X, y)
	final_nb_model.fit(X, y)
	final_rf_model.fit(X, y)

	# Reading the test data
	test_data = pd.read_csv("./dataset/Testing.csv").dropna(axis=1)

	test_X = test_data.iloc[:, :-1]
	test_Y = encoder.transform(test_data.iloc[:, -1])

	# Making prediction by take mode of predictions 
	# made by all the classifiers

	svm_preds = final_svm_model.predict(test_X)
	nb_preds = final_nb_model.predict(test_X)
	rf_preds = final_rf_model.predict(test_X)
	
	def custom_mode(lst):
		counts = Counter(lst)
		max_count = max(counts.values())
		modes = [k for k, v in counts.items() if v == max_count]
		return modes[0]  # Return the first mode

	try:
		final_preds = [custom_mode([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]
		#print(final_preds)
	except IndexError as e:
		print("Error occurred:", e)
		#print("Check which list might be causing the issue: svm_preds, nb_preds, or rf_preds")




	# final_preds = [mode([i,j,k])[0][0] for i,j,k in zip(svm_preds, nb_preds, rf_preds)]

	#print(f"Accuracy on Test dataset by the combined model\
	#: {accuracy_score(test_Y, final_preds)*100}")

	cf_matrix = confusion_matrix(test_Y, final_preds)
	symptoms = X.columns.values

	# Creating a symptom index dictionary to encode the
	# input symptoms into numerical form
	symptom_index = {}
	for index, value in enumerate(symptoms):
		symptom = " ".join([i.capitalize() for i in value.split("_")])
		symptom_index[symptom] = index

	data_dict = {
		"symptom_index":symptom_index,
		"predictions_classes":encoder.classes_
	}
	
	return data_dict,final_rf_model,final_nb_model,final_svm_model


data_dict,final_rf_model,final_nb_model,final_svm_model = predict()
preventive_measures = {
    'Paroxysmal Positional Vertigo': [
        "Avoid sudden head movements.",
        "Perform balance exercises.",
        "Use caution when changing positions, such as getting out of bed or looking up."
    ],
    'ais': [
        "Avoid sharing needles.",
        "Get tested regularly for HIV."
    ],
    'Acne': [
        "Keep skin clean by washing regularly.",
        "Avoid touching your face with dirty hands.",
        "Use non-comedogenic skincare products."
    ],
    'Alcoholic hepatitis': [
        "Limit alcohol consumption.",
        "Seek help for alcohol addiction.",
        "Maintain a healthy diet."
    ],
    'Allergy': [
        "Identify and avoid triggers.",
        "Keep indoor environments clean and dust-free.",
        "Use allergy medications as prescribed."
    ],
    'Arthritis': [
        "Maintain a healthy weight.",
        "Exercise regularly to keep joints flexible.",
        "Use joint protection techniques."
    ],
    'Bronchial Asthma': [
        "Avoid triggers such as smoke, pollen, and pet dander.",
        "Use inhalers as prescribed.",
        "Create an asthma action plan with your doctor."
    ],
    'Cervical Spondylosis': [
        "Maintain good posture.",
        "Take frequent breaks from sitting or standing.",
        "Perform neck-strengthening exercises."
    ],
    'Chicken pox': [
        "Get vaccinated.",
        "Avoid close contact with infected individuals.",
        "Practice good hygiene, such as frequent handwashing."
    ],
    'Chronic cholestasis': [
        "Manage underlying conditions like liver disease.",
        "Follow a healthy diet low in fat.",
        "Avoid alcohol consumption."
    ],
    'Common Cold': [
        "Wash hands frequently.",
        "Avoid close contact with sick individuals.",
        "Boost immune system with a healthy diet and regular exercise."
    ],
    'Dengue': [
        "Eliminate standing water where mosquitoes breed.",
        "Use mosquito repellents.",
        "Wear protective clothing, such as long sleeves and pants."
    ],
    'Diabetes': [
        "Maintain a healthy weight.",
        "Exercise regularly.",
        "Follow a balanced diet and monitor blood sugar levels."
    ],
    'Dimorphic hemmorhoids(piles)': [
        "Eat high-fiber foods.",
        "Stay hydrated.",
        "Avoid straining during bowel movements."
    ],
    'Drug Reaction': [
        "Take medications as prescribed by a healthcare professional.",
        "Inform healthcare providers of any known allergies.",
        "Be cautious when trying new medications."
    ],
    'Fungal infection': [
        "Keep skin clean and dry.",
        "Wear clean, breathable clothing.",
        "Avoid sharing personal items like towels and clothing."
    ],
    'GERD': [
        "Maintain a healthy weight.",
        "Avoid trigger foods, such as spicy or acidic foods.",
        "Eat smaller, more frequent meals."
    ],
    'Gastroenteritis': [
        "Practice good hygiene, such as frequent handwashing.",
        "Ensure food is properly cooked and stored.",
        "Avoid close contact with infected individuals."
    ],
    'Heart attack': [
        "Maintain a healthy diet low in saturated fats and cholesterol.",
        "Exercise regularly.",
        "Manage stress levels."
    ],
    'Hepatitis B': [
        "Get vaccinated.",
        "Practice safe sex and avoid sharing needles.",
        "Get tested for hepatitis B."
    ],
    'Hepatitis C': [
        "Avoid sharing needles.",
        "Practice safe sex.",
        "Get tested for hepatitis C."
    ],
    'Hepatitis D': [
        "Get vaccinated for hepatitis B.",
        "Avoid sharing needles.",
        "Practice safe sex."
    ],
    'Hepatitis E': [
        "Practice good hygiene, such as washing hands frequently.",
        "Drink clean, safe water.",
        "Avoid eating raw or undercooked shellfish."
    ],
    'Hypertension': [
        "Maintain a healthy weight.",
        "Exercise regularly.",
        "Follow a low-sodium diet."
    ],
    'Hyperthyroidism': [
        "Take prescribed medications as directed by a healthcare professional.",
        "Follow up with regular doctor appointments.",
        "Manage stress levels."
    ],
    'Hypoglycemia': [
        "Eat regular meals and snacks.",
        "Monitor blood sugar levels.",
        "Carry glucose tablets or snacks for emergencies."
    ],
    'Hypothyroidism': [
        "Take prescribed thyroid hormone replacement medications.",
        "Follow up with regular doctor appointments.",
        "Maintain a healthy diet and exercise regularly."
    ],
    'Impetigo': [
        "Keep skin clean and dry.",
        "Avoid scratching or picking at sores.",
        "Avoid close contact with infected individuals."
    ],
    'Jaundice': [
        "Practice good hygiene, such as frequent handwashing.",
        "Get vaccinated for hepatitis A and B.",
        "Avoid sharing personal items like razors and toothbrushes."
    ],
    'Malaria': [
        "Use mosquito nets while sleeping.",
        "Take prescribed antimalarial medications if traveling to affected areas.",
        "Use mosquito repellents."
    ],
    'Migraine': [
        "Identify and avoid triggers, such as certain foods or stress.",
        "Get regular sleep and maintain a consistent schedule.",
        "Manage stress levels."
    ],
    'Osteoarthristis': [
        "Maintain a healthy weight.",
        "Exercise regularly, including low-impact activities like swimming or cycling.",
        "Use assistive devices as needed."
    ],
    'Paralysis (brain hemorrhage)': [
        "Maintain a healthy lifestyle to prevent conditions like stroke.",
        "Exercise regularly.",
        "Manage chronic conditions like hypertension and diabetes."
    ],
    'Peptic ulcer disease': [
        "Avoid smoking.",
        "Limit alcohol consumption.",
        "Manage stress levels."
    ],
    'Pneumonia': [
        "Get vaccinated against pneumococcal pneumonia.",
        "Practice good hygiene, such as frequent handwashing.",
        "Avoid close contact with sick individuals."
    ],
    'Psoriasis': [
        "Moisturize skin regularly.",
        "Avoid trigger factors, such as stress or certain medications.",
        "Follow prescribed treatment plans."
    ],
    'Tuberculosis': [
        "Get vaccinated if not already vaccinated.",
        "Finish the full course of prescribed medications.",
        "Avoid close contact with infected individuals."
    ],
    'Typhoid': [
        "Practice good hygiene, such as frequent handwashing.",
        "Drink clean, safe water.",
        "Avoid eating raw or undercooked food."
    ],
    'Urinary tract infection': [
        "Stay hydrated.",
        "Practice good hygiene, such as wiping from front to back.",
        "Urinate after sexual activity."
    ],
    'Varicose veins': [
        "Maintain a healthy weight.",
        "Exercise regularly to improve circulation.",
        "Avoid sitting or standing for long periods."
    ],
    'Hepatitis A': [
        "Get vaccinated.",
        "Practice good hygiene, such as frequent handwashing.",
        "Avoid contaminated food and water."
    ]
}
def predictDisease(symptoms):
	#print("Symptomps : ",symptoms)
	symptoms = symptoms.split(",")
	
	# creating input data for the models
	input_data = [0] * len(data_dict["symptom_index"])
	# #print("Data dic",data_dict)
	# #print("Sym",symptoms)
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1
		
	# reshaping the input data and converting it
	# into suitable format for model predictions
	input_data = np.array(input_data).reshape(1,-1)


	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
	# final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
	final_prediction = np.unique([rf_prediction, nb_prediction, svm_prediction], return_counts=True)[0][0]
	if rf_prediction not in preventive_measures.keys():
		preventive_measures[rf_prediction] = ["Sorry, Unable to predict the symptoms"]
	predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction,
		"preventive_measures" : preventive_measures[rf_prediction]
	}
	return predictions
