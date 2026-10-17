import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DATA_PATH = os.path.join("data", "raw")

def fetch_delivery_data(data_path=RAW_DATA_PATH):
    """Descarga automáticamente train.csv desde Kaggle y lo extrae en data/raw/."""
    os.makedirs(data_path, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()
    
    dataset_name = "gauravmalik26/food-delivery-dataset"
    file_name = "train.csv"
    
    print(f"Iniciando descarga de '{file_name}'...")
    api.dataset_download_file(dataset_name, file_name=file_name, path=data_path)
    
    zip_path = os.path.join(data_path, f"{file_name}.zip")
    if os.path.exists(zip_path):
        print("Descomprimiendo archivo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(zip_path) 
        
    print("¡Ingesta completada!")

def load_delivery_data(data_path=RAW_DATA_PATH):
    """Carga el dataset en un DataFrame de Pandas."""
    csv_path = os.path.join(data_path, "train.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo {csv_path} no existe.")
    return pd.read_csv(csv_path)

if __name__ == "__main__":
    fetch_delivery_data()
    df = load_delivery_data()
    print("\n--- Vista previa ---")
    print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(df.head(2))