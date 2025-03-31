# Portfolio de Data Science

## Directorios del Proyecto:
# - churn_prediction/: Predicción de fuga de clientes
# - fake_news_detection/: Detección de noticias falsas
# - hotel_booking_cancellation/: Predicción de cancelaciones de reservas
# - resources/: Certificaciones y recursos adicionales

# Creación de estructura de carpetas en Python
import os

directories = [
    "churn_prediction", 
    "fake_news_detection", 
    "hotel_booking_cancellation", 
    "resources"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "README.md"), "w") as f:
        f.write(f"# {directory.replace('_', ' ').title()}\n\nDescripción del proyecto aquí.")

# Código para Churn Prediction
churn_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('churn_data.csv')
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, preds)}')
"""
with open("churn_prediction/churn_model.py", "w") as f:
    f.write(churn_code)

# Código para Fake News Detection
fake_news_code = """
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('fake_news.csv')
X = df['text'].values
y = df['label'].values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)
model = Sequential([
    Embedding(5000, 64, input_length=100),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_pad, y, epochs=5, batch_size=32, validation_split=0.2)
"""
with open("fake_news_detection/fake_news_model.py", "w") as f:
    f.write(fake_news_code)

# Código para Hotel Booking Cancellation
hotel_code = """
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('hotel_bookings.csv')
X = df.drop(columns=['is_canceled'])
y = df['is_canceled']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
"""
with open("hotel_booking_cancellation/hotel_model.py", "w") as f:
    f.write(hotel_code)

# Creación del README principal
with open("Portfolio_de_Data_Science.py", "w", encoding="utf-8") as f:
    f.write("""
#  Portfolio de Data Science

Este repositorio contiene proyectos destacados de Data Science, enfocados en Machine Learning, Deep Learning y Análisis de Datos.

##  Proyectos Incluidos:

1️ **Churn Prediction**: Predicción de fuga de clientes usando Random Forest, XGBoost y Redes Neuronales.
2️ **Fake News Detection**: Clasificación de noticias falsas con LSTM y Word Embeddings.
3️ **Hotel Booking Cancellation**: Predicción de cancelaciones de reservas con Redes Neuronales Multicapa y Dropout.

##  Contenido del Repositorio

- `churn_prediction/` → Proyecto de predicción de fuga de clientes.
- `fake_news_detection/` → Proyecto de detección de noticias falsas.
- `hotel_booking_cancellation/` → Predicción de cancelaciones de reservas.
- `resources/` → Certificaciones y materiales de aprendizaje.

##  Cómo Ejecutar los Proyectos

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/data-science-portfolio.git
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecutar los notebooks en cada carpeta.

##  Certificaciones en Data Science

- **Modelos Avanzados y Redes Neuronales** - Desafío Latam (Oct 2024) - ID: 36087
- **Machine Learning** - Desafío Latam (Jul 2024) - ID: 34112
- **Análisis Estadístico con Python** - Desafío Latam (May 2024) - ID: 31500
- **SQL para el Análisis de Datos** - Desafío Latam (Nov 2023) - ID: 26731

 **Autor:** [Marco Brante]  
 **GitHub:** [https://github.com/LordBrante](https://github.com/LordBrante)
    """)

# Creación del archivo de dependencias
with open("requirements.txt", "w") as f:
    f.write("""
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
nltk
    """)

print("Estructura del portafolio generada con éxito. ¡Sube los archivos a GitHub!")
