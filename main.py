import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Memuat dataset Boston Housing
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Menampilkan informasi singkat tentang dataset
print(data.head())
print(data.info())

# Memisahkan fitur dan target
X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Regresi Linear
model = LinearRegression()

# Melatih model
model.fit(X_train, y_train)

# Membuat prediksi pada data uji
predictions = model.predict(X_test)

# Menghitung error
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Visualisasi hasil prediksi
plt.scatter(y_test, predictions)
plt.xlabel("Harga Aktual")
plt.ylabel("Prediksi Harga")
plt.title("Hubungan antara Harga Aktual dan Prediksi Harga")
plt.show()
