import os
import random
import shutil
from PIL import Image, ImageEnhance

PASTA_BRUTA = r'C:\Users\Cliente Aspire\Desktop\DATASET_BRUTO'
PASTA_FINAL = r'C:\Users\Cliente Aspire\Desktop\DATASET_PRONTO'

QTD_TESTE = 50
META_TREINO = 1450

def augmentar_imagem(img):
    op = random.choice(['rotacao', 'espelhar', 'brilho', 'contraste', 'zoom'])

    if op == 'rotacao':
        return img.rotate(random.randint(-15, 15))

    if op == 'espelhar':
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    if op == 'brilho':
        fator = random.uniform(0.9, 1.1)
        return ImageEnhance.Brightness(img).enhance(fator)

    if op == 'contraste':
        fator = random.uniform(0.9, 1.1)
        return ImageEnhance.Contrast(img).enhance(fator)

    if op == 'zoom':
        w, h = img.size
        fator = random.uniform(0.05, 0.8)
        dx = int(w * fator)
        dy = int(h * fator)
        recorte = img.crop((dx, dy, w - dx, h - dy))
        return recorte.resize((w, h), Image.LANCZOS)

    return img


if not os.path.exists(PASTA_BRUTA):
    print("Pasta de dados não encontrada.")
else:
    for cidade in sorted(os.listdir(PASTA_BRUTA)):
        origem = os.path.join(PASTA_BRUTA, cidade)
        if not os.path.isdir(origem):
            continue

        treino = os.path.join(PASTA_FINAL, "Images", "train", cidade)
        teste = os.path.join(PASTA_FINAL, "Images", "test", cidade)

        os.makedirs(treino, exist_ok=True)
        os.makedirs(teste, exist_ok=True)

        imagens = [img for img in os.listdir(origem)
                   if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        random.shuffle(imagens)

        for i, img_nome in enumerate(imagens[:QTD_TESTE]):
            shutil.copy(
                os.path.join(origem, img_nome),
                os.path.join(teste, f"{cidade}_{i+1:04d}.jpg")
            )

        restantes = imagens[QTD_TESTE:]

        for i in range(META_TREINO):
            nome_saida = f"{cidade}_{i+1:04d}.jpg"

            if i < len(restantes):
                shutil.copy(
                    os.path.join(origem, restantes[i]),
                    os.path.join(treino, nome_saida)
                )
            else:
                base = random.choice(restantes)
                with Image.open(os.path.join(origem, base)) as img:
                    img = augmentar_imagem(img.convert("RGB"))
                    img.save(os.path.join(treino, nome_saida), quality=95)

        print(f"{cidade}")