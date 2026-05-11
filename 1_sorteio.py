import pandas as pd
import os

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
origem = os.path.join(desktop, "Borboletas_SS")
destino = os.path.join(desktop, "borboletas_sorteadas")

os.makedirs(destino, exist_ok=True)

for arquivo in os.listdir(origem):
    if arquivo.endswith(".csv"):
        caminho = os.path.join(origem, arquivo)

        try:
            dados = pd.read_csv(caminho)

            tamanho = min(len(dados), 1500)
            amostra = dados.sample(n=tamanho, random_state=42)

            amostra.to_csv(os.path.join(destino, arquivo), index=False)
            print(f"{arquivo}: {tamanho} registros selecionados")

        except Exception as erro:
            print(f"Problema em {arquivo}: {erro}")