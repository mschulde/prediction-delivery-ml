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
Para la ingesta de datos se creo el archivo `src/ingesta.py` con las siguientes responsabilidades:
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
Para cumplir con los principios de *duck typing* de Scikit-Learn (`fit`, `transform`, `fit_transform`) y permitir una integración sin fisuras en pipelines, se desarrollaron en `src/transformadores.py` dos clases modulares:

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

En `src/pipeline.py` se ensambló una arquitectura desacoplada mediante `ColumnTransformer` para procesar en paralelo las rutas de transformación:
### Prevención de Data Leakage
El cálculo de parámetros estadísticos (como las medianas de imputación, medias y desviaciones estándar del `StandardScaler`, y el vocabulario de categorías del `OneHotEncoder`) se calcula **únicamente con el método `.fit()` sobre el conjunto de entrenamiento**. Durante el procesamiento de nuevos datos o del test set, únicamente se ejecuta `.transform()`, garantizando cero contaminación informativa del conjunto de prueba.

---

## Fase 6: Entrenamiento, Métricas de Negocio, Validación y Serialización

### 6.1 Justificación de las Métricas de Evaluación (MAE vs. RMSE vs. $R^2$)

1. **Error Absoluto Medio (MAE - Métrica Primaria de Negocio):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \vert{}y_i - \hat{y}_i\vert{}$$
   - **Razón de Negocio:** Penaliza el error de forma puramente lineal y proporcional. Un desvío de 5 minutos equivale exactamente al doble de penalización que un desvío de 2.5 minutos. Es intuitiva, directamente auditable por los gerentes de logística y transferible al cálculo de costos operativos por retraso.
2. **Raíz del Error Cuadrático Medio (RMSE - Métrica de Penalización de Casos Críticos):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
   - **Razón Técnica:** Al elevar las diferencias al cuadrado, penaliza desproporcionadamente los errores grandes. Permite identificar si el modelo comete fallas intolerables en pedidos de larga duración.
3. **Coeficiente de Determinación ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
   - **Razón Estadística:** Mide qué porcentaje de la variabilidad del tiempo de entrega es explicada por las características procesadas respecto a una estimación ingenua basada en la media global.

### 6.2 A/B Testing Algorítmico y Resultados Empíricos
Para garantizar la mejor generalización en el entorno de producción, se implementó una estrategia de experimentación múltiple entrenando y evaluando tres arquitecturas de Machine Learning distintas:

| Arquitectura de Machine Learning | Error MAE (Test) | Varianza Explicada ($R^2$) | Estabilidad K-Fold (MAE Promedio $\pm$ STD) |
| :--- | :--- | :--- | :--- |
| **1. Regresión Lineal (Baseline)** | 4.98 min | 55.00 % | 4.99 min ($\pm 0.05$) |
| **2. Árbol de Decisión Simple** | 3.34 min | 79.45 % | 3.31 min ($\pm 0.03$) |
| **3. Random Forest Regressor** | **3.17 min** | **81.80 %** | **3.17 min** ($\pm 0.03$) |

### 6.3 Conclusión
Tras someter la arquitectura matemática a estrictas pruebas de generalización (Validación Cruzada K-Fold con 5 particiones aisladas), el equipo dictamina que el modelo **RANDOM FOREST** es el algoritmo definitivo y certificado para su despliegue a producción.

**Justificación Técnica de la Elección:**
1. **Minimización del Error Operativo:** Redujo el MAE en más de un 35% frente a la Regresión Lineal base, estableciendo el margen de equivocación logística en apenas 3.17 minutos.
2. **Capacidad de Comprensión No Lineal:** Alcanzó un $R^2$ cercano al 82%, demostrando la superioridad absoluta de los métodos de ensamble basados en árboles para descifrar interacciones logísticas complejas (como la sinergia negativa entre clima adverso, tráfico denso y vehículos en mal estado) que un modelo estrictamente paramétrico no pudo capturar.
3. **Estabilidad en Producción:** La desviación estándar transversal en el K-Fold fue de ínfimos $\pm 0.03$ minutos, demostrando científicamente que el Random Forest consolidó las reglas logísticas del negocio generalizando la información, sin caer en *Overfitting*.

### 6.4 Serialización MLOps para Producción (`models/`)
Utilizando la librería `joblib`, se serializó la tubería ganadora en la carpeta `models/`:
* `models/modelo_random_forest.joblib`: Este artefacto no es solo el estimador final, sino que **encapsula recursivamente toda la tubería `Pipeline`**. Contiene en su interior las clases personalizadas (`LimpiadorFormatosKaggle`, `CalculadorDistancia`), el `SimpleImputer`, el `StandardScaler`, el `OneHotEncoder` y los hiperparámetros del bosque. 

Esto permite que un microservicio o API de inferencia (e.g., FastAPI, Flask) cargue el archivo en memoria y procese pedidos JSON crudos entrantes en tiempo real, sin necesidad de reescribir la lógica de limpieza en el backend.

---

## Estructura del Proyecto y Guía de Reproducibilidad

### Estructura de Directorios
```text
prediction-delivery-ml/
│
├── data/
│   ├── raw/                      # Dataset crudo original descargado de Kaggle (train.csv)
│   └── processed/                # Particiones preprocesadas (train_processed.csv, test_processed.csv)
│
├── notebooks/
│   ├── 01_EDA_y_Separacion.ipynb               # Fase 3: Estratificación, Test ANOVA y Pearson
│   ├── 02_Pipeline_y_FeatureEngineering.ipynb   # Fase 4 y 5: Orquestación MLOps en memoria
│   ├── 03_Entrenamiento_RegresionLineal.ipynb   # Fase 6: Entrenamiento Modelo Paramétrico Base
│   ├── 04_Entrenamiento_ArbolDecision.ipynb     # Fase 6: Entrenamiento Modelo No Lineal (Baseline)
│   ├── 05_Entrenamiento_RandomForest.ipynb      # Fase 6: Entrenamiento Modelo de Ensamble (Campeón)
│   └── 06_Evaluacion_Final.ipynb               # Fase 6: Reporte Ejecutivo y Comparativa Visual
│
├── src/
│   ├── __init__.py
│   ├── ingesta.py                # Fase 2: Descarga y carga automatizada desde Kaggle
│   ├── transformadores.py        # Fase 4: POO (LimpiadorFormatosKaggle y CalculadorDistancia)
│   └── pipeline.py               # Fase 5: Tubería unificada de preprocesamiento preventivo de fugas
│
├── models/                       # Fase 6: Artefactos exportados listos para backend (.joblib)
├── requirements.txt              # Definición declarativa estricta de dependencias
├── .gitignore                    # Reglas de exclusión para credenciales y entornos virtuales
└── README.md                     # Documento Técnico del Proyecto (Entregable Final)