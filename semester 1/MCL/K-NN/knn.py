# 1. Baca data dari file CSV
import pandas as pd
takaran = pd.read_csv('takaran_saji.csv')
print("Data Awal:")
print(takaran)
# print(takaran.info()) 

# 2. Pisahkan fitur dan label
data = takaran.drop('klasifikasi', axis=1)
label = takaran['klasifikasi']
# print("Data dan Label:", data.head())
# print("Label:", label.head())

# 3. Bagi data menjadi data latih dan data uji
from sklearn.model_selection import train_test_split
data_latih, data_uji, label_latih, label_uji = train_test_split( data, label, test_size=0.2)
# print("Data Latih:", data_latih)
# print("Label Latih:", label_latih)
# print("Data Uji:", data_uji)
# print("Label Uji:", label_uji)

# 4. Standarisasi data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_latih)
data_latih = scaler.transform(data_latih)
data_uji = scaler.transform(data_uji)
# print("Data Latih setelah Standarisasi:", data_latih)
# print("Data Uji setelah Standarisasi:", data_uji)

# 5. Buat dan latih model KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(data_latih, label_latih)
prediksi = model.predict(data_uji)
# print("Prediksi Kelas untuk Data Uji:", prediksi)

# 6. Prediksi pada data uji
prediksi_probabiltas = model.predict_proba(data_uji)
# print("Probabilitas Prediksi untuk Data Uji:", prediksi_probabiltas)

# 7. Evaluasi hasil
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print(confusion_matrix(label_uji, prediksi))
print("Classification Report:")
print(classification_report(label_uji, prediksi, zero_division=0))