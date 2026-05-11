import pandas as pd
import requests
import os

base_csv = r'C:\Users\Cliente Aspire\Desktop\borboletas_sorteadas'
base_dataset = r'C:\Users\Cliente Aspire\Desktop\DATASET_BRUTO'

os.makedirs(base_dataset, exist_ok=True)

for nome_arquivo in sorted(os.listdir(base_csv)):
    if nome_arquivo.endswith(".csv"):
        cidade = nome_arquivo.replace(".csv", "")
        pasta_cidade = os.path.join(base_dataset, cidade)

        os.makedirs(pasta_cidade, exist_ok=True)

        dados = pd.read_csv(os.path.join(base_csv, nome_arquivo))

        print(f"{cidade}...")

        for i, link in enumerate(dados["identifier"]):
            try:
                nome_imagem = f"{cidade}_{i+1:04d}.jpg"
                caminho = os.path.join(pasta_cidade, nome_imagem)

                if not os.path.exists(caminho):
                    resposta = requests.get(link, timeout=15)

                    if resposta.status_code == 200:
                        with open(caminho, "wb") as arquivo:
                            arquivo.write(resposta.content)

            except:
                continue

print("processo finalizado")