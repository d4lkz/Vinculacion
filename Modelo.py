import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo Excel desde la ruta especificada
file_path = 'BD_Discapacidades.xlsx'
data = pd.read_excel(file_path)

# Seleccionar solo las columnas relevantes
columns_to_use = ['TIPO DE DISCAPACIDAD ', 'EDAD AÑOS/MESES', 'PORCENTAJE\n%', 'ASISTENCIA \nSI/NO']
data_cleaned = data[columns_to_use].copy()

data_cleaned['TIPO DE DISCAPACIDAD '] = data_cleaned['TIPO DE DISCAPACIDAD '].fillna('Sin Discapacidad')
data_cleaned['PORCENTAJE\n%'] = data_cleaned['PORCENTAJE\n%'].fillna('0%')
data_cleaned['PORCENTAJE\n%'] = data_cleaned['PORCENTAJE\n%'].replace('NO APLICA', '0%')

# Factorizar la columna 'TIPO DE DISCAPACIDAD '
factorized_values, unique_labels = pd.factorize(data_cleaned['TIPO DE DISCAPACIDAD '])

# Mostrar los valores únicos y su correspondencia con los números factorized
label_mapping = dict(enumerate(unique_labels))
print("Mapping de valores factorized:")
print(label_mapping)

# Convertir la columna 'TIPO DE DISCAPACIDAD ' a valores numéricos mediante codificación
data_cleaned.loc[:, 'TIPO DE DISCAPACIDAD '] = pd.factorize(data_cleaned['TIPO DE DISCAPACIDAD '])[0]

# Convertir valores no numéricos en la columna "PORCENTAJE\n%" a NaN
data_cleaned.loc[:, 'PORCENTAJE\n%'] = pd.to_numeric(data_cleaned['PORCENTAJE\n%'], errors='coerce')

# Manejar valores faltantes
data_cleaned.loc[:, 'TIPO DE DISCAPACIDAD '] = data_cleaned['TIPO DE DISCAPACIDAD '].fillna(data_cleaned['TIPO DE DISCAPACIDAD '].mode()[0])
data_cleaned.loc[:, 'EDAD AÑOS/MESES'] = data_cleaned['EDAD AÑOS/MESES'].fillna(data_cleaned['EDAD AÑOS/MESES'].median())
data_cleaned.loc[:, 'PORCENTAJE\n%'] = data_cleaned['PORCENTAJE\n%'].fillna(data_cleaned['PORCENTAJE\n%'].median())

# Verificar que todas las columnas sean numéricas
data_cleaned.info()

# Verificar la distribución de la columna objetivo en el conjunto de datos completo
distribution_full = data_cleaned['ASISTENCIA \nSI/NO'].value_counts()
print("Distribución en el conjunto completo:", distribution_full)

# Dividir los datos en conjuntos de entrenamiento y prueba con estratificación
X = data_cleaned.drop(columns=['ASISTENCIA \nSI/NO'])
y = data_cleaned['ASISTENCIA \nSI/NO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Verificar la distribución de la columna objetivo en los conjuntos de entrenamiento y prueba
distribution_train = y_train.value_counts()
distribution_test = y_test.value_counts()

print("Distribución en el conjunto de entrenamiento:", distribution_train)
print("Distribución en el conjunto de prueba:", distribution_test)

# Si los datos son adecuados, entrenar el modelo de Random Forest
if len(distribution_full) > 1 and len(distribution_train) > 1 and len(distribution_test) > 1:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(model, 'modelo_entrenado.pkl')

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:')
    print(conf_matrix)

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
else:
    print("No hay suficientes datos de ambas clases para entrenar el modelo.")
