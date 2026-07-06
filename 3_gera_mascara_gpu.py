import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamProcessor,
    SamModel,
)

BASE_DIR = "/home/samira/Área de trabalho/Experimentos Paper Sibgrapi 2026"

IMAGENS = os.path.join(BASE_DIR, "dataset_borboletas")
MASCARAS = os.path.join(BASE_DIR, "mascaras_binarias_sam")

#device = "cuda" if torch.cuda.is_available() else "cpu"
# DINO na GPU (se couber)
dino_device = "cpu"

# SAM na CPU
sam_device = "cpu"

# Localização por texto
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(dino_device)

dino_model.eval()   # <- aqui

# Segmentação
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(sam_device)

sam_model.eval()    # <- aqui


def criar_mascara(caminho_img, pasta_saida, nome):
    caminho_saida = os.path.join(pasta_saida, f"{nome}_mask.png")
    if os.path.exists(caminho_saida):
        return
    try:
        image_pil = Image.open(caminho_img).convert("RGB")
        w, h = image_pil.size

        # 1) GroundingDINO: localizar "butterfly"
        prompt = "butterfly."

        inputs = dino_processor(
            images=image_pil,
            text=prompt,
            return_tensors="pt"
        ).to(dino_device)

        with torch.inference_mode():
            outputs = dino_model(**inputs)

        results = dino_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=0.15,
            text_threshold=0.20,
            target_sizes=[(h, w)]
        )[0]

        boxes = results["boxes"]

        if len(boxes) == 0:
            print(f"Sem detecção: {nome}")
            mascara_vazia = Image.new("L", (w, h), 0)
            mascara_vazia.save(os.path.join(pasta_saida, f"{nome}_mask.png"))
            return

        # Pega a caixa com maior score
        scores = results["scores"]
        idx = torch.argmax(scores).item()
        box = boxes[idx].cpu().numpy().tolist()

        x1, y1, x2, y2 = box
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        # Verifica se a caixa é válida
        if x2 <= x1 or y2 <= y1:
            print(f"Caixa inválida: {nome}")

            mascara_vazia = Image.new("L", (w, h), 0)

            mascara_vazia.save(
                os.path.join(
                    pasta_saida,
                    f"{nome}_mask.png"
                )
            )
            return
        print(f"{nome}: box=({x1},{y1},{x2},{y2}), score={scores[idx].item():.3f}")
        input_box = [[[x1, y1, x2, y2]]]

        # 2) SAM: gerar máscara a partir da caixa
        sam_inputs = sam_processor(
            image_pil,
            input_boxes=input_box,
            return_tensors="pt"
        ).to(sam_device)

        with torch.inference_mode():
            sam_outputs = sam_model(**sam_inputs)

        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )[0]

        iou_scores = sam_outputs.iou_scores.cpu()[0, 0]
        best_mask_idx = torch.argmax(iou_scores).item()

        mask = masks[0, best_mask_idx].numpy()

        mascara = (mask * 255).astype(np.uint8)

        cv2.imwrite(caminho_saida, mascara)
        
        del image_pil, inputs, outputs, results, boxes, scores
        del sam_inputs, sam_outputs, masks, mask, mascara
        del input_box, iou_scores, best_mask_idx

    except Exception as erro:
        print(f"Erro em {nome}: {erro}")
    finally:
        import gc
        gc.collect()


if __name__ == "__main__":
    for tipo in ["treino", "teste"]:
        pasta_tipo = os.path.join(IMAGENS, tipo)

        if not os.path.exists(pasta_tipo):
            continue

        #print("Pastas encontradas:")
        #print(sorted(os.listdir(pasta_tipo)))
        #pasta_001 = os.path.join(IMAGENS, "treino", "001")

        #print("Existe?", os.path.exists(pasta_001))
        #print("É pasta?", os.path.isdir(pasta_001))
        #print("Arquivos em 001:", os.listdir(pasta_001)[:10])
        #print("Total:", len(os.listdir(pasta_001)))

        for cidade in sorted(os.listdir(pasta_tipo)):
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
