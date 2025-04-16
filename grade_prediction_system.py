import pandas as pd  # Veri işleme ve DataFrame kullanımı için
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test olarak ayırmak için
from sklearn.linear_model import LinearRegression  # Lineer regresyon modeli için
import matplotlib.pyplot as plt  # Grafik çizimi için

# Örnek veri setini oluşturuyoruz
data = {
    "gunluk_study_saati": [1, 2, 3, 4, 2, 1, 5, 3, 2.5, 4.5], 
    "devamsizlik_gunu": [10, 5, 2, 0, 7, 15, 1, 4, 6, 0], 
    "vize_notu": [50, 60, 70, 85, 55, 40, 95, 75, 65, 88],  
    "final_notu": [55, 65, 75, 90, 60, 45, 97, 78, 68, 92]
}

df = pd.DataFrame(data)  # Verileri pandas DataFrame yapısına dönüştürüyoruz

X = df[["gunluk_study_saati", "devamsizlik_gunu", "vize_notu"]]  
y = df["final_notu"]  # Hedef değişken (output)

# Veriyi eğitim ve test seti olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

# Lineer regresyon modelini oluşturuyoruz
model = LinearRegression()

# Modeli eğitim verisi ile eğitiyoruz 
model.fit(X_train, y_train)

# Model ile test verisi üzerinde tahmin yapıyoruz
y_pred = model.predict(X_test)

# Tahmin edilen sonuçları ekrana yazdırıyoruz
print(y_pred)

# Gerçek ve tahmin edilen değerleri görselleştiriyoruz
plt.plot(range(len(y_test)), y_test, label="Gerçek", marker="o")  # Gerçek değerler
plt.plot(range(len(y_pred)), y_pred, label="Tahmin", marker="x")  # Tahmin edilen değerler
plt.legend()  # Grafik açıklaması (etiketleri göster)
plt.title("Gerçek vs Tahmin Edilen Final Notu")  # Grafik başlığı
plt.xlabel("Örnek")  # X ekseni etiketi
plt.ylabel("Not")  # Y ekseni etiketi
plt.grid(True)  # Arka plana kılavuz çizgiler ekler
plt.tight_layout()  # Grafik üzerindeki yazıların taşmaması için otomatik hizalama
plt.show()  # Grafiği gösteriyoruz
