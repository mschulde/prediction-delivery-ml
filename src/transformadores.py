import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LimpiadorFormatosKaggle(BaseEstimator, TransformerMixin):
    """
    Transformador 1: Automatiza la limpieza de expresiones regulares y nulos
    que descubrimos manualmente durante la Fase 3 (EDA).
    """
    def __init__(self):
        pass # No necesitamos inicializar parámetros para este caso

    def fit(self, X, y=None):
        return self # fit solo devuelve self por la interfaz de Scikit-Learn

    def transform(self, X, y=None):
        # Creamos una copia para no alterar el dataset original en memoria
        X_ = X.copy()
        
        # 1. Limpieza numérica (Regex para extraer números)
        columnas_num_sucias = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries', 'Vehicle_condition']
        for col in columnas_num_sucias:
            if col in X_.columns:
                X_[col] = X_[col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
                X_[col] = pd.to_numeric(X_[col], errors='coerce')
                
        # 2. Limpieza categórica (Quitar espacios y estandarizar nulos)
        columnas_cat_clave = ['Road_traffic_density', 'Weatherconditions', 'Type_of_vehicle', 'Festival', 'City']
        for col in columnas_cat_clave:
            if col in X_.columns:
                X_[col] = X_[col].astype(str).str.strip()
                X_[col] = X_[col].replace(['NaN', 'nan', 'None', ''], np.nan)
                
        return X_


class CalculadorDistancia(BaseEstimator, TransformerMixin):
    """
    Transformador 2: Feature Engineering. 
    Convierte las 4 coordenadas inútiles en 1 atributo de alto impacto: Distancia_km.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        
        # Validamos que el dataset tenga las columnas espaciales
        cols_req = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
        if all(col in X_.columns for col in cols_req):
            
            # Convertir coordenadas a radianes para la matemática trigonométrica
            lat1 = np.radians(X_['Restaurant_latitude'])
            lon1 = np.radians(X_['Restaurant_longitude'])
            lat2 = np.radians(X_['Delivery_location_latitude'])
            lon2 = np.radians(X_['Delivery_location_longitude'])
            
            # Fórmula de Haversine
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371 # Radio medio de la Tierra en km
            
            # Crear la nueva característica
            X_['Distancia_km'] = c * r
            
            # Descartar las 4 variables originales (basado en el EDA)
            X_ = X_.drop(columns=cols_req)
            
        return X_