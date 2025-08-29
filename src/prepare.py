import pandas as pd
import json
import os

# Crear carpeta de salida si no existe
os.makedirs("data/interim", exist_ok=True)

# Rutas absolutas de los archivos
sample_path = "data/raw/sample_data.json"
train_path = "data/raw/train.json"
output_csv = "data/interim/train_plus_sample.csv"

# Leer sample_data.json
with open(sample_path, "r", encoding="utf-8") as f:
    sample_data = json.load(f)

# Leer train.json
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

# Convertir los diccionarios a listas de registros
sample_list = [dict(id=k, **v) for k, v in sample_data.items()]
train_list = [dict(id=k, **v) for k, v in train_data.items()]

# Crear DataFrames
df_sample = pd.DataFrame(sample_list)
df_train = pd.DataFrame(train_list)

# Unir los DataFrames
df_all = pd.concat([df_train], ignore_index=True)

# Guardar el resultado como CSV
output_path = "/home/daniel-linux/Universidad/" + output_csv
# Si hay columnas con listas o diccionarios, conviértelas a string
for col in df_all.columns:
    if df_all[col].apply(lambda x: isinstance(x, (list, dict))).any():
        df_all[col] = df_all[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
df_all.to_csv(output_path, index=False, sep = ';')

print(f"[prepare] Guardado {output_path}")