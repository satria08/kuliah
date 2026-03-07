import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Data
takaran = pd.read_csv('takaran_saji.csv')
data = takaran.drop('klasifikasi', axis=1)
label = takaran['klasifikasi']
print("Data: ")
print(data)

# Standarisasi
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Latih model dengan semua data
model = KNeighborsClassifier(n_neighbors=2)
model.fit(data_scaled, label)

# Prediksi data baru
data_baru = pd.DataFrame([[50, 3, 40]], columns=data.columns)
data_baru_scaled = scaler.transform(data_baru)
prediksi_baru = model.predict(data_baru_scaled)

print("Klasifikasi data [50, 3, 40]:", prediksi_baru[0])
