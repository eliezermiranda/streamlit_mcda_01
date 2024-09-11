# PROMETHEE II
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import streamlit as st



st.set_page_config(
    layout="wide",
    page_title="Decision App"
)

def page1():
    st.title("Main")

def page2():
    st.title("Dataset")
    excel_df

def results():
    st.title("Results")
    st.title("")
    st.write("Ranking:")
    st.write(ranked_matrix)

st.title('Decision App')
st.sidebar.title('Decision Model Parameters')

page = st.sidebar.selectbox("Select a page", ["Main", "Dataset"])

# Display the selected page
if page == "Main":
    page1()
elif page == "Dataset":
    page2()


# Reading the dataset
#excel_df = pd.read_excel('Dataset_PROMETHEE_XLSX_Excel.xlsx')



# File uploader widget in the sidebar
file =0
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (xlsx, csv)",
    type=["xlsx", "csv"]
)

# Check if a file has been uploaded
if uploaded_file is not None:
    # Determine the file type and read accordingly
    if uploaded_file.name.endswith('.xlsx'):
        excel_df = pd.read_excel(uploaded_file)
        file = 1
    elif uploaded_file.name.endswith('.csv'):
        # Try reading with comma separator first
        try:
            excel_df = pd.read_csv(uploaded_file)
            file = 1
        except pd.errors.ParserError:
            # If it fails, try reading with semicolon separator
            excel_df = pd.read_csv(uploaded_file, sep=';')
            file = 1
    
    # Display the dataframe
    st.write(excel_df)
else:
    st.write("Please upload a file.")








if file == 1:
    # Extracting the data from the dataset
    matriz_max_min = excel_df.iloc[1:, 0].tolist()
    matriz_pesos = excel_df.iloc[1:, 1].tolist()
    matriz_tipos = excel_df.iloc[1:, 2].tolist()
    matriz_indiferenca = excel_df.iloc[1:, 3].tolist()
    matriz_preferencia = excel_df.iloc[1:, 4].tolist()
    matriz_nomes_criterios = excel_df.iloc[1:, 5].tolist()
    matriz_decisao = excel_df.iloc[1:, 6:].values.tolist()
    matriz_nomes_alternativas = excel_df.iloc[0, 6:].tolist()
    n_colunas_alternativas = int(len(matriz_nomes_alternativas))
    n_linhas_criterios = int(len(matriz_nomes_criterios))





    # Aplication of the Max or Min function
    # Initialize a dictionary to store the matrices
    matrizes_app_max_min = {}

    # Loop to create and store each matrix for each criteria for Max or Min Function
    for g in range(n_linhas_criterios):
        # Create a matrix (in each iteraction)
        matriz_app_max_min = [[0 for _ in range(n_colunas_alternativas)] for _ in range(n_colunas_alternativas)]
        
        # Store each matrix in the dictionary with a unique key
        matrizes_app_max_min[f"matriz_app_max_min_{g}"] = matriz_app_max_min



    # Loop(s) to apply the functions max or min and save in the auxiliar matrix "matriz_app_max_min_{i+1}" and save in the dictionary "matrizes_app_max_min"
    for i in range(n_linhas_criterios):
        matriz_app_max_min = [[0 for _ in range(n_colunas_alternativas)] for _ in range(n_colunas_alternativas)]
        for j in range(n_colunas_alternativas):
            linha = [[0 for _ in range(n_colunas_alternativas)] for _ in range(1)]
            for k in range(n_colunas_alternativas):
                app_max_min: float = 0
                if matriz_max_min[i] == "Max":
                    app_max_min = matriz_decisao[i][j] - matriz_decisao[i][k]
                elif matriz_max_min[i] =="Min":
                    app_max_min = matriz_decisao[i][k] - matriz_decisao[i][j]
                linha [0][k] = app_max_min
            # Append the row to the matrix "matriz_app_max_min"
            for l in range(n_colunas_alternativas):
                matriz_app_max_min[j][l] = linha[0][l]
        # Store the matrix in the dictionary with a unique key
        matrizes_app_max_min[f"matriz_app_max_min_{i}"] = matriz_app_max_min



    g = int(0)
    i = int(0)
    j = int(0)
    k = int(0)
    l = int(0)





    # Preference Function
    # Initialize a dictionary to store the matrices
    matrizes_app_preference = {}


    # Loop to create and store each matrix
    for g in range(n_linhas_criterios):
        # Create a matrix (in each iteraction)
        matriz_app_preference = [[0 for _ in range(n_colunas_alternativas)] for _ in range(n_colunas_alternativas)]
        # Store each matrix in the dictionary with a unique key
        matrizes_app_preference[f"matriz_app_preference_{g}"] = matriz_app_preference




    # Loop(s) to apply the preference functions and save in the auxiliar matrix "matriz_app_preference_{i+1}" and save in the dictionary "matrizes_app_preference"
    q_indiference = float()
    p_preference = float()
    for i in range(n_linhas_criterios):
        matriz_app_max_min = matrizes_app_max_min[f"matriz_app_max_min_{i}"]
        matriz_app_preference = [[0 for _ in range(n_colunas_alternativas)] for _ in range(n_colunas_alternativas)]
        for j in range(n_colunas_alternativas):
            linha = [[0 for _ in range(n_colunas_alternativas)] for _ in range(1)]
            for k in range(n_colunas_alternativas):
                app_preference: float = 0
                if matriz_tipos[i] == 1:
                    print("Função de preferência Tipo 1")
                    # Definir caminho provisório 
                elif matriz_tipos[i] == 2:
                    print("Função de preferência Tipo 2")
                    # Definir caminho provisório 
                elif matriz_tipos[i] == 3:
                    print("Função de preferência Tipo 3")
                    # Definir caminho provisório 
                elif matriz_tipos[i] == 4:
                    print("Função de preferência Tipo 4")
                    # Definir caminho provisório 
                elif matriz_tipos[i] == 5:
                    app_max_min = float(matriz_app_max_min[j][k])
                    q_indiference = float(matriz_indiferenca[i])
                    p_preference = float(matriz_preferencia[i])
                    p_peso = float(matriz_pesos[i])
                    if app_max_min <= q_indiference:
                        app_preference = 0
                    elif app_max_min > p_preference:
                        app_preference = 1.0
                    else:
                        app_preference = ((app_max_min-q_indiference)/(p_preference-q_indiference))
                elif matriz_tipos[i] == 6:
                    print("Função de preferência Tipo 6")
                    # Definir caminho provisório
                linha [0][k] = app_preference * p_peso / 100
            # Append the row to the matrix "matriz_app_preference"
            for l in range(n_colunas_alternativas):
                matriz_app_preference[j][l] = linha[0][l]
        # Store the matrix in the dictionary with a unique key
        matrizes_app_preference[f"matriz_app_preference_{i}"] = matriz_app_preference



    matriz_app_average_degree = []
    matriz_app_average_degree = [[0 for _ in range(n_colunas_alternativas)] for _ in range(n_colunas_alternativas)]
    linha = [[0 for _ in range(n_colunas_alternativas)] for _ in range(1)]


    # Loop(s) to obtain the "degree average" for later calculus or the Phi+ and Phi-
    for j in range(n_colunas_alternativas):
        for k in range(n_colunas_alternativas):
            app_average_degree: float = 0
            for m in range(n_linhas_criterios):
                matriz_app_preference = matrizes_app_preference[f"matriz_app_preference_{m}"]
                app_average_degree = app_average_degree + matriz_app_preference[j][k]
            linha [0][k] = app_average_degree/n_linhas_criterios
        # Append the row to the matrix "matriz_app_average_degree"
        for l in range(n_colunas_alternativas):
            matriz_app_average_degree[j][l] = linha[0][l]




    # Phi+ Calculus
    matriz_Phi_plus = [[0] * 1 for _ in range(n_colunas_alternativas)]
    for n in range(n_colunas_alternativas):
        Phi_plus: float = 0
        for o in range(n_colunas_alternativas):
            Phi_plus = Phi_plus + matriz_app_average_degree[n][o]
        matriz_Phi_plus[n][0] = Phi_plus



    # Phi- Calculus
    matriz_Phi_minus = [[0] * 1 for _ in range(n_colunas_alternativas)]
    for n in range(n_colunas_alternativas):
        Phi_minus: float = 0
        for o in range(n_colunas_alternativas):
            Phi_minus = Phi_minus + matriz_app_average_degree[o][n]
        matriz_Phi_minus[n][0] = Phi_minus



    # Phi Calculus
    matriz_Phi = [[0] * 1 for _ in range(n_colunas_alternativas)]
    Phi: float = 0
    for n in range(n_colunas_alternativas):
        Phi_plus = matriz_Phi_plus[n][0]
        Phi_minus = matriz_Phi_minus[n][0]
        Phi = Phi_plus - Phi_minus
        matriz_Phi[n][0] = Phi


    #Ranking
    matriz_Phi_np = np.array(matriz_Phi)
    flattened = matriz_Phi_np.flatten()
    sorted_indices = np.argsort(-flattened)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(flattened) + 1)
    ranked_matrix = ranks.reshape(matriz_Phi_np.shape)






    results()



