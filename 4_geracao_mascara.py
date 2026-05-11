import os
import cv2
from ultralytics import YOLO
from PIL import Image
from rembg import remove

BASE_DIR = r'C:\Users\Cliente Aspire\Desktop\DATASET_PRONTO_KAGGLE'
IMAGENS = os.path.join(BASE_DIR, 'Images')
MASCARAS = os.path.join(BASE_DIR, 'Masks')

model = YOLO("yolov8n.pt")


def criar_mascara(caminho_img, pasta_saida, nome):
    try:
        imagem = cv2.imread(caminho_img)

        resultado = model.predict(
            source=caminho_img,
            conf=0.15,
            classes=[75],
            verbose=False
        )

        if len(resultado[0].boxes) > 0:
            x1, y1, x2, y2 = map(int, resultado[0].boxes[0].xyxy[0].cpu().numpy())
            recorte = imagem[y1:y2, x1:x2]
        else:
            recorte = imagem

        img_pil = Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        img_sem_fundo = remove(img_pil)

        mascara = img_sem_fundo.split()[3]

        caminho_saida = os.path.join(pasta_saida, f"{nome}_mask.png")
        mascara.resize((224, 224), Image.LANCZOS).save(caminho_saida)

    except Exception as erro:
        print(f"Erro em {nome}: {erro}")


if __name__ == "__main__":
    for tipo in ["train", "test"]:
        pasta_tipo = os.path.join(IMAGENS, tipo)
        if not os.path.exists(pasta_tipo):
            continue

        for cidade in os.listdir(pasta_tipo):
            caminho_cidade = os.path.join(pasta_tipo, cidade)
            if not os.path.isdir(caminho_cidade):
                continue

            pasta_saida = os.path.join(MASCARAS, tipo, cidade)
            os.makedirs(pasta_saida, exist_ok=True)

            print(f"Gerando máscaras em {tipo}/{cidade}...")

            for arquivo in os.listdir(caminho_cidade):
                if arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
                    criar_mascara(
                        os.path.join(caminho_cidade, arquivo),
                        pasta_saida,
                        os.path.splitext(arquivo)[0]
                    )

