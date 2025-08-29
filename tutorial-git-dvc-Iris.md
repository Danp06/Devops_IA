
# Iris + DVC (v2): **Mismo nombre de archivo**, cambios de datos y control de versiones

Objetivo: usar **el mismo nombre** `data/raw/iris.csv` aunque el contenido cambie con el tiempo, y **versionar** cada actualización con **DVC + Git**. Además, **subir métricas y cambios** al repositorio tras el **primer `dvc repro`** y en cada actualización.

---

## 1) Preparar entorno

```bash
mkdir iris-dvc && cd iris-dvc

git init
dvc init

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install dvc scikit-learn pandas matplotlib joblib
```

Configura un **remoto** (ejemplo local en un directorio hermano `../dvcstore`):

```bash
dvc remote add -d localstore ../dvcstore
git add .dvc/config
git commit -m "Configure DVC remote (local)"
```

---

## 2) Dataset IRIS **(v1)** — **mismo nombre**

Descarga la primera versión y guárdala con **el mismo nombre** que usarás siempre:

```bash
mkdir -p data/raw
curl -L -o data/raw/iris.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
```

Rastrea con DVC (crea `data/raw/iris.csv.dvc`):

```bash
echo "data/raw/*" >> .gitignore # ignorar el archivo de datos en git
dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc .gitignore
git commit -m "Track Iris v1 at data/raw/iris.csv"
dvc push     # sube los datos al remoto
```

> Nota: El **nombre** `data/raw/iris.csv` **no cambia**. Lo que cambia es el **contenido**, y DVC guarda un hash distinto en cada versión del archivo `.dvc`.

---

## 3) Pipeline y parámetros

`dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw/iris.csv
      - src/prepare.py
    outs:
      - data/interim/iris_clean.csv

  train:
    cmd: python src/train.py --params params.yaml
    deps:
      - data/interim/iris_clean.csv
      - src/train.py
      - params.yaml
    outs:
      - models/model.pkl
    metrics:
      - metrics/train.json:
          cache: false
```

`params.yaml`:

```yaml
seed: 42
train:
  test_size: 0.2
  n_estimators: 100
  max_depth: 3
```

`src/prepare.py`:

```python
import pandas as pd, os
os.makedirs("data/interim", exist_ok=True)
df = pd.read_csv("data/raw/iris.csv").dropna()
df.to_csv("data/interim/iris_clean.csv", index=False)
print("[prepare] Guardado data/interim/iris_clean.csv")
```

`src/train.py`:

```python
import pandas as pd, yaml, json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/interim/iris_clean.csv")
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["train"]["test_size"], random_state=params["seed"]
)

model = RandomForestClassifier(
    n_estimators=params["train"]["n_estimators"],
    max_depth=params["train"]["max_depth"],
    random_state=params["seed"]
)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

os.makedirs("metrics", exist_ok=True)
with open("metrics/train.json", "w") as f:
    json.dump({"accuracy": acc}, f, indent=2)

print("[train] Accuracy:", acc)
```

Crea la estructura:

```bash
mkdir -p src data/interim models metrics
# crea los archivos anteriores con tu editor
```

---

## 4) **Primer run** y **subida de métricas/cambios**

```bash
dvc repro
```

Ahora **versiona en Git** los metadatos y **sube datos y métricas**:

```bash
git add dvc.yaml dvc.lock params.yaml metrics/train.json
git commit -m "First run: build pipeline, model & metrics"
dvc push     # asegura que outs pesados (datos/modelos) están en el remoto
git push     # sube metadatos al repositorio git
```

> Recomendación: después de **cada** `dvc repro`, hacer siempre:  
> `git add ... && git commit -m "mensaje"` → `dvc push` → `git push`

---

## 5) **Actualizar datos** manteniendo **el mismo nombre** (crear **v2**)

Supongamos que llega una nueva versión de datos (más filas/ajustes). **Sobrescribe** el **mismo archivo** y **fuerza** a DVC a recalcular su hash:

```bash
# sobrescribe el mismo path con nuevos contenidos
curl -L -o data/raw/iris.csv https://example.com/iris_v2.csv

# actualiza el tracking de DVC para el MISMO archivo
dvc add -f data/raw/iris.csv

git add data/raw/iris.csv.dvc
git commit -m "Iris v2 (same path: data/raw/iris.csv)"
dvc push
```

Vuelve a **reproducir** y **sube métricas/cambios**:

```bash
dvc repro
git add dvc.lock metrics/train.json
git commit -m "Retrain with Iris v2; update metrics"
dvc push
git push
```

> Clave: El **archivo** en Git que cambia es **`data/raw/iris.csv.dvc`** (no el CSV). Ese `.dvc` guarda el nuevo **hash** del contenido **v2**.

---

## 6) Cambios de **parámetros** (nueva versión de experimento)

```bash
# editar params.yaml
# train:
#   n_estimators: 200
#   max_depth: 5

dvc repro
git add params.yaml dvc.lock metrics/train.json
git commit -m "Tune: n_estimators=200, max_depth=5"
dvc push
git push
```

Compara métricas entre commits:

```bash
dvc metrics diff HEAD~1
```

---

## 7) **Cambiar entre versiones** del **mismo archivo raw**

Para **volver a una versión anterior** de los datos (mismo nombre, distinto contenido), usa **Git** para moverte entre commits y **DVC** para materializar el contenido correcto:

```bash
# ver historial de la pista DVC del archivo
git log -- data/raw/iris.csv.dvc

# cambiar a un commit anterior
git checkout <commit-antiguo>
dvc checkout   # trae al workspace la versión de datos correspondiente al .dvc de ese commit
dvc pull       # si fuese necesario descargar desde el remoto
```

Para regresar al último estado:

```bash
git switch -    # o git checkout <main|master>
dvc checkout
dvc pull
```

> También puedes comparar datasets entre revisiones:  
> `dvc diff <rev1> <rev2> --targets data/raw/iris.csv`

---

## 8) Recuperar en otra máquina

```bash
git clone <repo>
cd iris-dvc
dvc pull
dvc repro
```

---

## 9) Checklist por run

```bash
# 1. Cambios en datos (mismo nombre) o código/params
#    - si datos cambiaron: dvc add -f data/raw/iris.csv

# 2. Reproducir
dvc repro

# 3. Versionar y subir
git add dvc.lock params.yaml metrics/train.json data/raw/iris.csv.dvc
git commit -m "Exp: datos/params actualizados"
dvc push
git push
```
