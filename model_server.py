
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle

weather_data = pd.read_csv('weather_data_final.csv')
RX_train, RX_test, Ry_train, Ry_test = train_test_split(
    weather_data[['Temperature', 'Humidity', 'Pressure']], 
    weather_data['Rain'], test_size=0.25, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(RX_train, Ry_train)
new_data = pd.DataFrame([[25, 70, 1010]])  # Example new data point

classification = rf_classifier.predict(new_data)
if classification == 1:
    print("Rain is likely.")
else:
    print("No rain is likely.")

# Calculate evaluation metrics
accuracy = accuracy_score(Ry_test, rf_classifier.predict(RX_test))
precision = precision_score(Ry_test, rf_classifier.predict(RX_test), average='weighted')
recall = recall_score(Ry_test, rf_classifier.predict(RX_test), average='weighted')
f1 = f1_score(Ry_test, rf_classifier.predict(RX_test), average='weighted')

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

with open('random_forest_classifier.pkl', 'wb') as file:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle

weather_data = pd.read_csv('weather_data_final.csv')
RX_train, RX_test, Ry_train, Ry_test = train_test_split(
    weather_data[['Temperature', 'Humidity', 'Pressure']], 
    weather_data['Rain'], test_size=0.25, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(RX_train, Ry_train)
new_data = pd.DataFrame([[25, 70, 1010]])  # Example new data point

classification = rf_classifier.predict(new_data)
if classification == 1:
    print("Rain is likely.")
else:
    print("No rain is likely.")

# Calculate evaluation metrics
accuracy = accuracy_score(Ry_test, rf_classifier.predict(RX_test))
precision = precision_score(Ry_test, rf_classifier.predict(RX_test), average='weighted')
recall = recall_score(Ry_test, rf_classifier.predict(RX_test), average='weighted')
f1 = f1_score(Ry_test, rf_classifier.predict(RX_test), average='weighted')

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

with open('random_forest_classifier.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)