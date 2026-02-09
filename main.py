import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# doc file
data = pd.read_csv('weather_forecast_data.csv')
# print(data.head())
# data.info()

# chia data
N, d = data.shape
x = data.iloc[:, 0:d-1]
y = data.iloc[:, d-1].values.reshape(-1,1)
# print(x)

# Mã hóa nhãn (Label Encoding)
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra + chuẩn hóa dữ liệu
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# 3. HUẤN LUYỆN MÔ HÌNH (Logistic Regression)
model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)

# 4. DỰ ĐOÁN VÀ ĐÁNH GIÁ
y_pred = model.predict(x_test)

# Tính độ chính xác
acc = accuracy_score(y_test, y_pred)
print(f"Độ chính xác (Accuracy): {acc * 100:.2f}%")
# y_pred = le.inverse_transform(y_pred)  # Chuyển đổi nhãn dự đoán về dạng ban đầu
# print(f"Dự đoán: {y_pred}")

print(classification_report(y_test, y_pred, target_names=le.classes_))


def predict_new_weather():

    print("\nVui lòng nhập các thông số thời tiết:")
    
    try:
        # Nhập các thông số
        temperature = float(input("  - Nhiệt độ (Temperature, °C): "))
        humidity = float(input("  - Độ ẩm (Humidity, %): "))
        wind_speed = float(input("  - Tốc độ gió (Wind Speed, km/h): "))
        cloud_cover = float(input("  - Độ che phủ mây (Cloud Cover, %): "))
        pressure = float(input("  - Áp suất (Pressure, hPa): "))
        
        # Tạo DataFrame với tên cột giống dữ liệu gốc
        new_data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Wind_Speed': [wind_speed],
            'Cloud_Cover': [cloud_cover],
            'Pressure': [pressure]
        })
        
        # Chuẩn hóa dữ liệu mới (sử dụng scaler đã fit từ tập train)
        new_data_scaled = sc.transform(new_data)
        
        # Dự đoán
        prediction = model.predict(new_data_scaled)
        prediction_proba = model.predict_proba(new_data_scaled)
        
        # Chuyển đổi kết quả về dạng ban đầu
        result = le.inverse_transform(prediction)[0]
        
        # Hiển thị kết quả
        print("\n" + "-"*60)
        print("KẾT QUẢ DỰ ĐOÁN:")
        print("-"*60)
        print(f"Dự đoán: {result.upper()}")
        print(f"Xác suất 'No Rain': {prediction_proba[0][0]:.2%}")
        print(f"Xác suất 'Rain': {prediction_proba[0][1]:.2%}")
        print("-"*60)
        
        return result, prediction_proba
        
    except ValueError:
        print("Lỗi: Vui lòng nhập số hợp lệ!")
        return None, None
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        return None, None
predict_new_weather()