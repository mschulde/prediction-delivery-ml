# Proyecto Machine Learning End-to-End: Predicción de Tiempo de Entrega (Food Delivery)

> **Documento Técnico y Reporte de Ingeniería**
> **Asignatura:** Aprendizaje Supervisado  
> **Integrantes:** Juan Palacios, Sebastian Chulde, Fausto Liquinchana
> **Actividad Práctica # 1:** Diseño e Implementación de un Pipeline de Machine Learning End-to-End  
> **Stack Tecnológico:** Python 3.11+, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib, Kaggle API  

---

## Tabla de Contenidos
1. [Fase 1: Encuadre del Negocio y Objetivos Estratégicos](#fase-1-encuadre-del-negocio-y-objetivos-estratégicos)
2. [Fase 2: Ingesta Automatizada de Datos](#fase-2-ingesta-automatizada-de-datos)
3. [Fase 3: Separación Estratificada, Prevención de Data Snooping y EDA](#fase-3-separación-estratificada-prevención-de-data-snooping-y-eda)
4. [Fase 4: Limpieza de Datos y Feature Engineering Personalizado](#fase-4-limpieza-de-datos-y-feature-engineering-personalizado)
5. [Fase 5: Pipeline Modular de Transformación (MLOps)](#fase-5-pipeline-modular-de-transformación-mlops)
6. [Fase 6: Entrenamiento, Métricas de Negocio, Validación y Serialización](#fase-6-entrenamiento-métricas-de-negocio-validación-y-serialización)
7. [Estructura del Proyecto y Guía de Reproducibilidad](#estructura-del-proyecto-y-guía-de-reproducibilidad)

---

## Fase 1: Encuadre del Negocio y Objetivos Estratégicos

### 1.1 Definición del Problema
En plataformas de entrega bajo demanda (ej. aplicaciones de *Food Delivery* como Rappi, UberEats o PedidosYa), la incertidumbre en los tiempos de despacho genera fricción operativa, sobrecostos por compensación de pedidos retrasados y degradación en la experiencia del usuario final.

El objetivo principal de este proyecto es **desarrollar un sistema de Machine Learning end-to-end capaz de predecir de forma cuantitativa y exacta el tiempo total de entrega en minutos (`Time_taken(min)`)**, desde la recolección del pedido hasta la entrega física al cliente, utilizando variables operativas, ambientales, geográficas y del repartidor.

### 1.2 Alineación Estratégica con Procesos de Negocio
El modelo implementado impacta directamente en los siguientes objetivos corporativos y operacionales:

1. **Optimización del Despacho y Asignación Dinámica:** Alimenta algoritmos downstream de asignación y enrutamiento, permitiendo despachar al repartidor óptimo en el momento exacto y evitar colas de espera en los comercios aliados.
2. **Reducción de Márgenes de Error y Cancelaciones:** Una estimación precisa reduce discrepancias con el *Estimated Time of Arrival* (ETA) mostrado en la aplicación, minimizando reclamos de soporte, compensaciones económicas y cancelaciones tempranas de usuarios.
3. **Optimización de Capacidad y Recursos:** Facilita al usuario predecir picos de sobrecarga en función de condiciones climáticas adversas, festividades y congestión de tráfico para balancear tarifas dinámicas e incentivos a repartidores.

### 1.3 Formulación Formal del Algoritmo
* **Paradigma de Aprendizaje:** **Supervisado**. Se dispone de un histórico etiquetado donde cada observación cuenta con el tiempo real de entrega obtenido en campo ($y$).
* **Tipo de Tarea Predictiva:** **Regresión Univariada**. La variable objetivo $y \in \mathbb{R}^+$ representa un valor continuo y escalar correspondiente a los minutos de entrega. Las variables explicativas componen una matriz multivariada de características $X \in \mathbb{R}^{n \times d}$.
* **Estrategia de Aprendizaje (Batch vs. Online Learning):** Se selecciona **Aprendizaje Batch (Offline)** por las siguientes razones:
  - Los patrones estructurales del tráfico urbano, clima y geografía cambian de forma gradual y no milisegundo a milisegundo.
  - El volumen de datos (~45k observaciones) y su inferencia en producción se ajustan óptimamente a ciclos periódicos de reentrenamiento programado (semanal o quincenal).

---

## Fase 2: Ingesta Automatizada de Datos

### 2.1 Selección de la Base de Datos
Se seleccionó el repositorio de datos abierto **Food Delivery Dataset** alojado en [Kaggle](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset), el cual consolida **45,593 registros y 20 columnas originales** con telemetría espacial, atributos de pedidos, calificaciones de repartidores e información situacional.

### 2.2 Función de Ingesta Reproducible (`src/ingesta.py`)
Para la ingesta de datos se creo el archivo [`src/ingesta.py`](file:///D:/Catolica/prediction-delivery-ml/src/ingesta.py) con las siguientes responsabilidades:
* `fetch_delivery_data()`: Se autentica de forma programática con la API oficial de Kaggle (`KaggleApi`), descarga el archivo empaquetado `train.csv.zip`, lo descomprime automáticamente en `data/raw/train.csv` y purga los artefactos temporales.
* `load_delivery_data()`: Valida la existencia física del archivo crudo y lo carga a memoria en un DataFrame de Pandas bajo manejo de excepciones.

---

## Fase 3: Separación Estratificada, Prevención de Data Snooping y EDA

### 3.1 Prevención de Data Snooping Bias y Estrategia de Muestreo
El sesgo de indagación (*Data Snooping Bias*) se produce cuando el científico de datos explora, normaliza o toma decisiones algorítmicas utilizando la totalidad del conjunto de datos antes de separarlo, introduciendo involuntariamente información del futuro o del test set en el proceso de modelado (*Data Leakage*).

Para blindar la generalización real:
1. **Separación Inmediata:** La partición 80% entrenamiento (36,474 filas) y 20% prueba (9,119 filas) se realiza como primer paso estricto.
2. **Muestreo Estratificado (`StratifiedShuffleSplit`):** Se identificó que la densidad de tráfico (`Road_traffic_density`) es una variable crítica que altera sustancialmente la varianza del tiempo de entrega. Una partición puramente aleatoria arriesgaba subrepresentar condiciones extremas (como atascos intensos `Jam` o tráfico bajo `Low`). Con la estratificación, la distribución original se replicó de forma idéntica en ambos conjuntos:
   - *Low:* 33.80%
   - *Jam:* 31.06%
   - *Medium:* 24.03%
   - *High:* 9.71%
   - *NaN / Missing:* 1.40%
3. **Persistencia Aislada:** Los subconjuntos se exportaron a `data/processed/train_processed.csv` y `test_processed.csv`, garantizando que todo el EDA subsecuente se ejecute **exclusivamente sobre el conjunto de entrenamiento**.

### 3.2 Análisis Exploratorio de Datos (EDA) y Hallazgos Clave

#### A. Pruebas de Hipótesis Categóricas (ANOVA Unidireccional)
Se evaluó si las medias del tiempo de entrega difieren significativamente entre las categorías mediante el estadístico $F$ de ANOVA ($p\text{-value} < 10^{-5}$ en todas):
* **`Festival` ($F\text{-Score} \approx 3350$):** Es la variable categórica más determinante. Durante festivales o eventos masivos, los tiempos de entrega sufren un salto drástico hacia arriba.
* **`Road_traffic_density` ($F\text{-Score} \approx 2768$):** La progresión de tiempos escala monótonamente de `Low` $\rightarrow$ `Medium` $\rightarrow$ `High` $\rightarrow$ `Jam`.
* **`City` ($F\text{-Score} \approx 1118$):** Las zonas semi-urbanas y metropolitanas exhiben dinámicas de tiempo marcadamente distintas a las áreas urbanas compactas.
* **`Weatherconditions` ($F\text{-Score} \approx 403$):** Condiciones meteorológicas severas (tormentas de arena, niebla densa y lluvia) deprimen la velocidad del despacho.
* **`Type_of_vehicle` ($F\text{-Score} \approx 341$):** El tipo de transporte (motocicletas vs. bicicletas/scooters) condiciona la agilidad del trayecto.

#### B. Análisis de Correlación Lineal (Pearson)
En las variables continuas y ordinales numéricas sobre el conjunto de entrenamiento:
* **`multiple_deliveries` ($r = +0.39$):** Fuerte correlación positiva; cada parada intermedia adicional penaliza sustancialmente el tiempo acumulado.
* **`Delivery_person_Ratings` ($r = -0.34$):** Correlación negativa clara; los repartidores con mayores calificaciones operan con mayor rapidez y eficiencia en la ruta.
* **`Delivery_person_Age` ($r = +0.30$):** Correlación positiva moderada; los repartidores jóvenes muestran tiempos ligeramente menores de desplazamiento.
* **`Vehicle_condition` ($r = -0.24$):** Correlación negativa; vehículos en estado degradado (nivel 0) registran mayores retrasos mecánicos y operativos.

---

## Fase 4: Limpieza de Datos y Feature Engineering Personalizado

### 4.1 Tratamiento de Valores Faltantes
A través del EDA se detectaron valores ausentes codificados como cadenas de texto sucias (`"NaN"`, `"None"`, `" "`) y valores faltantes reales:
* **Estrategia Numérica:** Imputación por la **mediana** (`SimpleImputer(strategy='median')`), protegiendo la imputación de distorsiones provocadas por valores atípicos (*outliers*) o colas asimétricas.
* **Estrategia Categórica:** Imputación por la **moda / categoría más frecuente** (`SimpleImputer(strategy='most_frequent')`), manteniendo la coherencia semántica en densidades de tráfico y factores climáticos.

### 4.2 Transformadores Personalizados con la Interfaz Scikit-Learn
Para cumplir con los principios de *duck typing* de Scikit-Learn (`fit`, `transform`, `fit_transform`) y permitir una integración sin fisuras en pipelines, se desarrollaron en [`src/transformadores.py`](file:///D:/Catolica/prediction-delivery-ml/src/transformadores.py) dos clases modulares:

#### 1. `LimpiadorFormatosKaggle`
* **Justificación de Negocio:** El dataset contenía inconsistencias de captura de datos (números embebidos en texto, cadenas de texto con espacios colaterales).
* **Operación:** Extrae componentes numéricos válidos en `Delivery_person_Age`, `Delivery_person_Ratings`, `multiple_deliveries` y `Vehicle_condition` vía expresiones regulares `r'(\d+\.?\d*)'`, y estandariza tokens de texto vacíos a `np.nan`.

#### 2. `CalculadorDistancia`
* **Justificación Matemática y de Negocio:** Cuatro coordenadas geográficas aisladas (`Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`) no guardan una relación lineal interpretable con el tiempo. El determinante físico real del tiempo de traslado es la **distancia de desplazamiento**.
* **Fundamento Físico-Matemático:** Implementa la **Fórmula de Haversine**, calculando la distancia ortodrómica sobre una esfera de radio medio $R = 6371\text{ km}$:

$$\Delta\phi = \phi_2 - \phi_1, \quad \Delta\lambda = \lambda_2 - \lambda_1$$

$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$

$$c = 2 \cdot \arcsin\left(\sqrt{a}\right)$$

$$\text{Distancia\_km} = R \cdot c$$

* **Reducción de Dimensionalidad:** Al generar `Distancia_km`, el transformador descarta de inmediato las cuatro columnas de coordenadas crudas, mitigando la multicolinealidad y reduciendo el espacio de búsqueda del estimador.

---

## Fase 5: Pipeline Modular de Transformación (MLOps)

En [`src/pipeline.py`](file:///D:/Catolica/prediction-delivery-ml/src/pipeline.py) se ensambló una arquitectura desacoplada mediante `ColumnTransformer` para procesar en paralelo las rutas de transformación:
### Prevención de Data Leakage
El cálculo de parámetros estadísticos (como las medianas de imputación, medias y desviaciones estándar del `StandardScaler`, y el vocabulario de categorías del `OneHotEncoder`) se calcula **únicamente con el método `.fit()` sobre el conjunto de entrenamiento**. Durante el procesamiento de nuevos datos o del test set, únicamente se ejecuta `.transform()`, garantizando cero contaminación informativa del conjunto de prueba.

---

## Fase 6: Entrenamiento, Métricas de Negocio, Validación y Serialización

### 6.1 Justificación de las Métricas de Evaluación (MAE vs. RMSE vs. $R^2$)

1. **Error Absoluto Medio (MAE - Métrica Primaria de Negocio):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   - **Razón de Negocio:** Penaliza el error de forma puramente lineal y proporcional. Un desvío de 5 minutos equivale exactamente al doble de penalización que un desvío de 2.5 minutos. Es intuitiva, directamente auditable por los gerentes de logística y transferible al cálculo de costos operativos por retraso.
2. **Raíz del Error Cuadrático Medio (RMSE - Métrica de Penalización de Casos Críticos):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   - **Razón Técnica:** Al elevar las diferencias al cuadrado, penaliza desproporcionadamente los errores grandes. Permite identificar si el modelo comete fallas intolerables en pedidos de larga duración.
3. **Coeficiente de Determinación ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
   - **Razón Estadística:** Mide qué porcentaje de la variabilidad del tiempo de entrega es explicada por las características procesadas respecto a una estimación ingenua basada en la media global.

### 6.2 Resultados Empíricos y Análisis de Overfitting
Se entrenó el algoritmo de **Regresión Lineal (`LinearRegression`)**, arrojando el siguiente comportamiento cuantitativo:

| Instancia de Evaluación               | Métrica MAE (Minutos)     | Coeficiente $R^2$         | Interpretación Técnica                                    |
| ------------------------------------- | ------------------------- | ------------------------- | ----------------------------------------------- | 
| **Conjunto de Entrenamiento (Train)** | **4.97 min**              | **0.5543**                | Ajuste base del estimador                       |    
| **Conjunto de Prueba (Test Set)**     | **4.98 min**              | **0.5500**                | Generalización en datos no vistos               |     
| **Diferencia (Train - Test)**         | **0.01 min**              | **0.0043**                | **Ausencia total de sobreajuste (Overfitting)** |  
| **Validación Cruzada ($K=5$ Folds)**  | **4.99 min** ($\pm 0.05$)  | **0.5537** ($\pm 0.0084$) | Estabilidad matemática uniforme                  |     

> **Veredicto de Generalización:** La desviación estándar de apenas $\pm 0.05$ minutos en la validación cruzada y la diferencia de 0.01 minutos entre Train y Test demuestran que el modelo mantiene un comportamiento determinista y generalizable, inmune a perturbaciones del muestreo.

### 6.3 Diagnóstico Visual de Residuos
La evaluación de los residuos ($e_i = y_i - \hat{y}_i$) reveló:
* **Campana Gaussiana de Errores:** La distribución de residuos exhibe una distribución normal centrada en $0.0$, demostrando que los errores son homocedásticos y de naturaleza estocástica, sin sesgos sistemáticos de subestimación ni sobreestimación.
* **Correlación Predicción vs. Realidad:** La nube de dispersión sigue fielmente la directriz diagonal de 45° (*Línea de la Perfección*).

### 6.4 Serialización MLOps para Producción (`models/`)
Utilizando la librería `joblib`, se serializaron los componentes desacoplados del sistema en la carpeta `models/`:
* `models/pipeline_preprocesamiento.joblib`: Pipeline ajustado con los pesos, medianas y codificadores.
* `models/modelo_regresion_lineal.joblib`: Estimador entrenado con los coeficientes óptimos.

Esto permite que un microservicio o API de inferencia (e.g., FastAPI, Flask) cargue ambos componentes en memoria y procese pedidos entrantes en tiempo real con latencia inferior a 15 ms.

---

## Estructura del Proyecto y Guía de Reproducibilidad

### Estructura de Directorios
```text
prediction-delivery-ml/
│
├── data/
│   ├── raw/                      # Dataset crudo original descargado de Kaggle (train.csv)
│   └── processed/                # Particiones train_processed.csv y test_processed.csv
│
├── notebooks/
│   ├── 01_EDA_y_Separacion.ipynb             # Fase 3: Muestreo estratificado, EDA (ANOVA y Pearson)
│   ├── 02_Pipeline_y_FeatureEngineering.ipynb # Fase 4 y 5: Ensayos de transformadores y ColumnTransformer
│   └── 03_Entrenamiento_Modelo.ipynb         # Fase 6: Entrenamiento, K-Fold CV, métricas y serialización
│
├── src/
│   ├── __init__.py
│   ├── ingesta.py                # Fase 2: Descarga y carga automatizada desde Kaggle
│   ├── transformadores.py        # Fase 4: Clases LimpiadorFormatosKaggle y CalculadorDistancia
│   └── pipeline.py               # Fase 5: Tubería unificada de preprocesamiento
│
├── models/                       # Fase 6: Modelos y pipelines serializados en formato .joblib
├── requirements.txt              # Definición declarativa de dependencias
├── .gitignore                    # Reglas de exclusión para datos procesados y credenciales
└── README.md                     # Documento técnico del proyecto (Entregable 3.1)
```

### Instrucciones de Instalación y Ejecución

#### 1. Clonar el repositorio y preparar el entorno virtual
```bash
# Clonar repositorio
git clone <URL_DEL_REPOSITORIO>
cd prediction-delivery-ml

# Crear y activar entorno virtual
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

# Instalar dependencias del sistema
pip install -r requirements.txt
```

#### 2. Configuración de Credenciales de Kaggle
Coloque su archivo `kaggle.json` en la ruta correspondiente del sistema:
* **Windows:** `C:\Users\<Usuario>\.kaggle\kaggle.json`
* **Linux/macOS:** `~/.kaggle/kaggle.json`

O defina las variables de entorno:
```bash
export KAGGLE_USERNAME="su_usuario"
export KAGGLE_KEY="su_api_key"
```

#### 3. Ejecución de la Ingesta Automatizada
```bash
python src/ingesta.py
```

#### 4. Ejecución del Flujo de Trabajo
Los notebooks se ejecutan de manera secuencial para reproducir las fases analíticas y de entrenamiento:
```bash
jupyter notebook
```
1. `notebooks/01_EDA_y_Separacion.ipynb` (Particiona y exporta los datos a `data/processed/`).
2. `notebooks/02_Pipeline_y_FeatureEngineering.ipynb` (Valida las transformaciones en memoria).
3. `notebooks/03_Entrenamiento_Modelo.ipynb` (Entrena, valida y exporta los artefactos a `models/`).
