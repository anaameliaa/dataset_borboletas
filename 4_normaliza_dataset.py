from pathlib import Path
from PIL import Image

# Ajuste estas pastas
PASTA_ORIGINAIS = Path("borboletas_novo_original/dataset_final")
PASTA_MASCARAS = Path("mascaras_binarias_sam")

# Pastas de saída
SAIDA_ORIGINAIS = Path("borboletas_novo_original/dataset_final_normalizado")
SAIDA_MASCARAS = Path("mascaras_normalizado")

TAMANHO = (224, 224)

EXTENSOES = [".jpg", ".jpeg", ".png", ".webp"]


def redimensionar_imagem(caminho_entrada, caminho_saida):
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(caminho_entrada) as img:
        img = img.convert("RGB")
        img = img.resize(TAMANHO, Image.LANCZOS)
        img.save(caminho_saida.with_suffix(".jpg"), quality=95)


def redimensionar_mascara(caminho_entrada, caminho_saida):
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(caminho_entrada) as mask:
        mask = mask.convert("L")
        mask = mask.resize(TAMANHO, Image.NEAREST)

        # garante máscara binária: preto ou branco
        mask = mask.point(lambda p: 255 if p > 127 else 0)

        mask.save(caminho_saida.with_suffix(".png"))


def processar_pasta(pasta_entrada, pasta_saida, tipo):
    total = 0
    erros = 0

    for arquivo in pasta_entrada.rglob("*"):
        if not arquivo.is_file():
            continue

        if arquivo.suffix.lower() not in EXTENSOES:
            continue

        relativo = arquivo.relative_to(pasta_entrada)
        destino = pasta_saida / relativo

        try:
            if tipo == "imagem":
                redimensionar_imagem(arquivo, destino)
            else:
                redimensionar_mascara(arquivo, destino)

            total += 1

            if total % 500 == 0:
                print(f"{tipo}: {total} arquivos processados")

        except Exception as e:
            erros += 1
            print(f"Erro em {arquivo}: {e}")

    print(f"\n{tipo} finalizado.")
    print(f"Total processado: {total}")
    print(f"Erros: {erros}")


print("Processando imagens originais...")
processar_pasta(PASTA_ORIGINAIS, SAIDA_ORIGINAIS, "imagem")

print("\nProcessando máscaras...")
processar_pasta(PASTA_MASCARAS, SAIDA_MASCARAS, "mascara")

print("\nTudo finalizado.")
print("Imagens 224:", SAIDA_ORIGINAIS)
print("Máscaras 224:", SAIDA_MASCARAS)
