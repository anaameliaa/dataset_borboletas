import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance

PASTA_BRUTA = Path("borboletas_novo_original/imagens")
PASTA_FINAL = Path("borboletas_novo_original/dataset_final")

QTD_TESTE = 50
META_TREINO = 1500



OPERACOES = ["rotacao", "espelhar", "brilho", "contraste", "zoom"]

random.seed(42)

def sortear_transformacao_usada(base_name, usados):
    tentativas = 0

    while tentativas < 1000:
        op = random.choice(OPERACOES)

        if op == "rotacao":
            parametro = random.randint(-20, 20)
            if parametro == 0:
                tentativas += 1
                continue

        elif op == "espelhar":
            parametro = "horizontal"

        elif op == "brilho":
            parametro = round(random.uniform(0.85, 1.15), 2)

        elif op == "contraste":
            parametro = round(random.uniform(0.85, 1.15), 2)

        elif op == "zoom":
            parametro = round(random.uniform(0.05, 0.15), 2)

        assinatura = (base_name, op, parametro)

        if assinatura not in usados:
            usados.add(assinatura)
            return op, parametro

        tentativas += 1

    return None, None
    
    

def augmentar_imagem(img, op, parametro):
    if op == "rotacao":
        return img.rotate(parametro)

    if op == "espelhar":
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    if op == "brilho":
        return ImageEnhance.Brightness(img).enhance(parametro)

    if op == "contraste":
        return ImageEnhance.Contrast(img).enhance(parametro)

    if op == "zoom":
        w, h = img.size
        dx = int(w * parametro)
        dy = int(h * parametro)
        recorte = img.crop((dx, dy, w - dx, h - dy))
        return recorte.resize((w, h), Image.LANCZOS)

    return img


if not PASTA_BRUTA.exists():
    print("Pasta de dados não encontrada:", PASTA_BRUTA)

else:
    for pasta_especie in sorted(PASTA_BRUTA.iterdir()):
        if not pasta_especie.is_dir():
            continue

        especie = pasta_especie.name

        treino = PASTA_FINAL /"treino" / especie
        teste = PASTA_FINAL /"teste" / especie

        treino.mkdir(parents=True, exist_ok=True)
        teste.mkdir(parents=True, exist_ok=True)

        imagens = [
            img for img in pasta_especie.iterdir()
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        random.shuffle(imagens)

        imagens_teste = imagens[:QTD_TESTE]
        imagens_treino = imagens[QTD_TESTE:]

        for i, img_path in enumerate(imagens_teste):
            shutil.copy(img_path, teste / f"{especie}_{i + 1:04d}.jpg")

        imagens_treino_reais = imagens_treino[:META_TREINO]
        treino_reais_original = len(imagens_treino_reais)

        for i, img_path in enumerate(imagens_treino_reais):
            shutil.copy(img_path, treino / f"{especie}_{i + 1:04d}.jpg")

        usados = set()
        bases_augmentation = imagens_treino_reais.copy()
        proximo_indice = treino_reais_original + 1

        while proximo_indice <= META_TREINO and imagens_treino_reais:
            base = random.choice(imagens_treino_reais)

            op, parametro = sortear_transformacao_usada(base.name, usados)

            if op is None:
                print(f"Não foi possível gerar nova transformação para {base.name}")
                continue

            nome_saida = f"{especie}_{proximo_indice:04d}.jpg"

            try:
                with Image.open(base) as img:
                    img = img.convert("RGB")
                    img_aug = augmentar_imagem(img, op, parametro)
                    img_aug.save(treino / nome_saida, quality=95)

                proximo_indice += 1

            except Exception as e:
                print(f"Erro ao augmentar {base}: {e}")

        print(
            f"{especie}: teste={len(imagens_teste)}, "
            f"treino_reais={treino_reais_original}, "
            f"treino_final={proximo_indice - 1}"
        )
