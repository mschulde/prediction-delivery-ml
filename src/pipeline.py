from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def construir_pipeline_preprocesamiento():
    """
    Construye la tubería final para imputar nulos y escalar/codificar variables.
    """
    num_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Vehicle_condition', 'Distancia_km']
    cat_cols = ['Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']

    num_pipeline = Pipeline([
        ('imputador', SimpleImputer(strategy='median')),
        ('escalador', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputador', SimpleImputer(strategy='most_frequent')),
        ('codificador', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocesador_final = ColumnTransformer([
        ('numericos', num_pipeline, num_cols),
        ('categoricos', cat_pipeline, cat_cols)
    ])

    return preprocesador_final