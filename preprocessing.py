import pandas as pd

def preprocess(pathname):
    file_path = pathname
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    matrix_data = []
    for line in lines[1:]:
        values = line.split('\t')[1:]
        float_values = [float(value.replace(',', '.').strip()) for value in values]
        matrix_data.append(float_values)

    distance_matrix_cleaned = pd.DataFrame(matrix_data)
    return distance_matrix_cleaned
