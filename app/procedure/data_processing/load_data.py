import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None