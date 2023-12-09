Не удалось подобрать подходящий файл, поэтому это просто общий случай

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# 1 Загрузка данных из CSV файла
data = pd.read_csv('data.csv')

# Подготовка данных

# Разделение данных на тренировочный, валидационный и тестовый наборы
train_data = data[:800]
val_data = data[800:900]
test_data = data[900:]

# Преобразование данных в формат, подходящий для использования в модели RNN
X_train = train_data.drop(columns=['target']).values
y_train = train_data['target'].values

X_val = val_data.drop(columns=['target']).values
y_val = val_data['target'].values

X_test = test_data.drop(columns=['target']).values
y_test = test_data['target'].values

#2 Создание сети
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(X_train.shape[1],)))
model.add(Dense(units=1))

# 3 Обучение модели
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

# 4 Оценка производительности
loss, mae = model.evaluate(X_test, y_test)
print("Loss: ", loss)
print("MAE: ", mae)



