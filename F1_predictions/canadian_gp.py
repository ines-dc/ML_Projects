import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache("F1_cache")
session_2024 = fastf1.get_session(2024, 3, "R")
session_2024.load()

laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"]= laps_2024["LapTime"].dt.total_seconds()

qualifying_2025= pd.DataFrame({
    "Driver": ["George Russell","Max Verstappen","Oscar Piastri","Andrea Kimi Antonelli","Lewis Hamilton","Fernando Alonso","Lando Norris","Charles Leclerc","Isack Hadjar","Alexander Albon","Yuki Tsunoda","Franco Colapinto","Nico Hülkenberg","Oliver Bearman","Esteban Ocon","Gabriel Bortoleto","Carlos Sainz Jr.","Lance Stroll","Liam Lawson","Pierre Gasly"],
    "QualifyingTime (s)": [70.899,71.059,71.120, 71.391,   71.526,  71.586,71.625,71.682, 71.867,71.907,72.102, 72.142,   72.183,72.340, 72.634,72.385,72.398, 72.517,72.525,72.667]
})

driver_mapping = {
    "Lando Norris": "NOR",
    "Oscar Piastri": "PIA",
    "Max Verstappen": "VER",
    "George Russell": "RUS",
    "Andrea Kimi Antonelli": "ANT",
    "Lewis Hamilton": "HAM",
    "Charles Leclerc": "LEC",
    "Liam Lawson": "LAW",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Pierre Gasly": "GAS",
    "Jack Doohan": "DOO",
    "Esteban Ocon": "OCO",
    "Oliver Bearman": "BEA",
    "Isack Hadjar": "HAD",
    "Yuki Tsunoda": "TSU",
    "Carlos Sainz Jr.": "SAI",
    "Alexander Albon": "ALB",
    "Nico Hülkenberg": "HUL",
    "Gabriel Bortoleto": "BOR"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

merge_data= qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

X = merge_data[["QualifyingTime (s)"]]
y = merge_data[["LapTime (s)"]]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check Data source")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
model.fit(X_train,y_train)

predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

print("\n Predicted 2025 Canadian GP Winner is: \n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

#Evaluate Model
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred)} seconds")