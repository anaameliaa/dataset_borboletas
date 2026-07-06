import random
import re
import warnings
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", InsecureRequestWarning)

PASTA_ORIGEM = Path("borboletas_novo_original")
PASTA_IMAGENS = PASTA_ORIGEM / "imagens"
ARQUIVO_CONTROLE = Path("selecionadas_download.csv")

ABAS = ["GBIF", "iNaturalist"]
N_POR_ESPECIE = 1550

MAX_WORKERS = 32
TAMANHO_LOTE = 200
TIMEOUT = 30

random.seed(42)
PASTA_IMAGENS.mkdir(parents=True, exist_ok=True)

LICENCAS_PROBLEMATICAS = {
    "",
    "nan",
    "none",
    "null",
    "all rights reserved",
    "todos os direitos reservados",
}


def normalizar_licenca(valor):
    if pd.isna(valor):
        return ""
    return str(valor).strip().lower()


def licenca_permitida(licenca):
    lic = normalizar_licenca(licenca)
    return not (
        lic in LICENCAS_PROBLEMATICAS
        or "all rights reserved" in lic
        or "todos os direitos reservados" in lic
    )


def limpar_id(valor):
    valor = str(valor).strip()
    if valor.endswith(".0"):
        valor = valor[:-2]
    return valor


def consultar_licenca_gbif(gbif_id):
    gbif_id = limpar_id(gbif_id)

    if not gbif_id or gbif_id.lower() == "nan":
        return ""

    try:
        r = requests.get(
            f"https://api.gbif.org/v1/occurrence/{gbif_id}",
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            return ""

        data = r.json()
        return data.get("license") or data.get("rights") or ""

    except Exception:
        return ""


def obter_id_item(row, aba):
    if aba == "GBIF":
        if "gbifID" in row.index:
            return limpar_id(row["gbifID"])
        if "id" in row.index:
            return limpar_id(row["id"])
        return limpar_id(row.iloc[0])

    if aba == "iNaturalist":
        if "id" in row.index:
            return limpar_id(row["id"])
        if "image_url" in row.index:
            return str(row["image_url"]).strip()
        return limpar_id(row.iloc[0])

    return limpar_id(row.iloc[0])


def obter_url(row, aba):
    if aba == "GBIF":
        if "identifier" in row.index:
            return str(row["identifier"]).strip()
        if "image_url" in row.index:
            return str(row["image_url"]).strip()
        return str(row.iloc[0]).strip()

    if aba == "iNaturalist":
        if "image_url" in row.index:
            return str(row["image_url"]).strip()
        if "url" in row.index:
            return str(row["url"]).strip()
        return str(row.iloc[0]).strip()

    return str(row.iloc[0]).strip()


def montar_url_inat_photo(photo_id, tamanho="original", ext="jpeg"):
    return f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{tamanho}.{ext}"


def obter_imagem_de_observacao_inat(url):
    try:
        obs_id = url.rstrip("/").split("/")[-1]
        api_url = f"https://api.inaturalist.org/v1/observations/{obs_id}"

        r = requests.get(
            api_url,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 dataset-download-script"},
        )

        if r.status_code != 200:
            return None

        data = r.json()
        results = data.get("results", [])

        if not results:
            return None

        photos = results[0].get("photos", [])
        if not photos:
            return None

        photo_id = photos[0].get("id")
        if photo_id:
            return montar_url_inat_photo(photo_id, tamanho="original")

        photo_url = photos[0].get("url")
        if photo_url:
            return (
                photo_url
                .replace("/square.", "/original.")
                .replace("/medium.", "/original.")
            )

        return None

    except Exception:
        return None


def normalizar_url(url):
    url = str(url).strip()
    url_lower = url.lower()

    if "inaturalist.org/observations/" in url_lower:
        image_url = obter_imagem_de_observacao_inat(url)
        if image_url:
            return image_url

    m = re.search(r"inaturalist\.org/photos/(\d+)", url_lower)
    if m:
        return montar_url_inat_photo(m.group(1))

    m = re.search(r"/photos/(\d+)/", url_lower)
    if "inaturalist" in url_lower and m:
        return montar_url_inat_photo(m.group(1))

    if "boldsystems.org/pics/" in url_lower:
        return url.replace("http://", "https://")

    return url


def extensao_da_url(url):
    path = urlparse(str(url)).path.lower()

    if path.endswith(".jpeg") or path.endswith(".jpg"):
        return ".jpg"
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".webp"):
        return ".webp"
    if path.endswith(".gif"):
        return ".gif"

    return ".jpg"


def baixar_imagem(url, caminho_saida):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
            "Gecko/20100101 Firefox/120.0"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7",
        "Connection": "keep-alive",
    }

    try:
        r = requests.get(
            url,
            headers=headers,
            timeout=TIMEOUT,
            stream=True,
            allow_redirects=True,
            verify=False,
        )

        if r.status_code != 200:
            return f"erro_status_{r.status_code}"

        content_type = r.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            return "erro_html_em_vez_de_imagem"

        with open(caminho_saida, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if not caminho_saida.exists() or caminho_saida.stat().st_size == 0:
            return "erro_arquivo_vazio"

        return "ok"

    except Exception as e:
        return f"erro_{e}"


def tentar_download(url_download, original_url, caminho_saida):
    status = baixar_imagem(url_download, caminho_saida)

    if status == "ok":
        return status, url_download, False

    if "inaturalist-open-data.s3.amazonaws.com/photos/" in url_download:
        url_medium = url_download.replace("/original.", "/medium.")
        status_medium = baixar_imagem(url_medium, caminho_saida)

        if status_medium == "ok":
            return status_medium, url_medium, True

    if url_download != original_url:
        status_original = baixar_imagem(original_url, caminho_saida)

        if status_original == "ok":
            return status_original, original_url, True

    if caminho_saida.exists() and caminho_saida.stat().st_size == 0:
        caminho_saida.unlink()

    return status, url_download, False


def carregar_controle():
    if ARQUIVO_CONTROLE.exists() and ARQUIVO_CONTROLE.stat().st_size > 0:
        return pd.read_csv(ARQUIVO_CONTROLE)

    return pd.DataFrame(
        columns=[
            "arquivo",
            "aba",
            "linha_excel",
            "id_item",
            "filename",
            "original_url",
            "download_url",
            "licenca",
            "status",
            "fallback_usado",
        ]
    )


def baixar_tarefa(tarefa):
    status, url_final, fallback_usado = tentar_download(
        tarefa["url_download"],
        tarefa["original_url"],
        tarefa["tmp_path"],
    )

    return {
        **tarefa,
        "status": status,
        "url_final": url_final,
        "fallback_usado": fallback_usado,
    }


def processar_lote(lote, controle, registros_especie, pasta_especie, especie, arquivo, baixadas):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(baixar_tarefa, tarefa) for tarefa in lote]

        for future in as_completed(futures):
            r = future.result()

            if baixadas >= N_POR_ESPECIE:
                if r["tmp_path"].exists():
                    r["tmp_path"].unlink()
                continue

            if r["status"] == "ok":
                baixadas += 1

                filename = f"{especie}_{baixadas:04d}{r['ext']}"
                final_path = pasta_especie / filename
                r["tmp_path"].rename(final_path)

                registro = {
                    "arquivo": r["arquivo"],
                    "aba": r["aba"],
                    "linha_excel": r["linha_excel"],
                    "id_item": r["id_item"],
                    "filename": filename,
                    "original_url": r["original_url"],
                    "download_url": r["url_final"],
                    "licenca": r["licenca"],
                    "status": "ok",
                    "fallback_usado": r["fallback_usado"],
                }

                registros_especie.append(registro)
                controle = pd.concat([controle, pd.DataFrame([registro])], ignore_index=True)

                if baixadas % 100 == 0:
                    print(f"{arquivo}: {baixadas}/{N_POR_ESPECIE} baixadas")

            else:
                if r["tmp_path"].exists():
                    r["tmp_path"].unlink()

                registro = {
                    "arquivo": r["arquivo"],
                    "aba": r["aba"],
                    "linha_excel": r["linha_excel"],
                    "id_item": r["id_item"],
                    "filename": "",
                    "original_url": r["original_url"],
                    "download_url": r["url_final"],
                    "licenca": r["licenca"],
                    "status": r["status"],
                    "fallback_usado": r["fallback_usado"],
                }

                controle = pd.concat([controle, pd.DataFrame([registro])], ignore_index=True)

    return controle, registros_especie, baixadas


controle = carregar_controle()
resumo = []

for i in range(1, 27):
    especie = f"{i:03d}"
    arquivo = f"{especie}.xlsx"
    caminho = PASTA_ORIGEM / arquivo
    pasta_especie = PASTA_IMAGENS / especie
    pasta_especie.mkdir(parents=True, exist_ok=True)

    if not caminho.exists():
        print(f"Arquivo não encontrado: {caminho}")
        continue

    ja_ok = controle[
        (controle["arquivo"] == arquivo)
        & (controle["status"] == "ok")
    ]

    if len(ja_ok) >= N_POR_ESPECIE:
        print(f"{arquivo}: já tem {len(ja_ok)} imagens OK. Pulando.")
        resumo.append({
            "arquivo": arquivo,
            "baixadas": len(ja_ok),
            "tentadas": 0,
            "candidatas": 0,
        })
        continue

    print(f"\nProcessando {arquivo}...")

    xls = pd.ExcelFile(caminho)
    candidatas = []

    for aba in ABAS:
        if aba not in xls.sheet_names:
            continue

        df = pd.read_excel(caminho, sheet_name=aba)
        df = df.copy()

        df["__aba"] = aba
        df["__linha_excel"] = range(2, len(df) + 2)
        df["__id_item"] = df.apply(lambda row: obter_id_item(row, aba), axis=1)
        df["__url"] = df.apply(lambda row: obter_url(row, aba), axis=1)

        usadas = controle[
            (controle["arquivo"] == arquivo)
            & (controle["aba"] == aba)
        ]

        ids_usados = set(usadas["id_item"].astype(str))
        linhas_usadas = set(usadas["linha_excel"].astype(str))

        df = df[
            ~df["__id_item"].astype(str).isin(ids_usados)
            & ~df["__linha_excel"].astype(str).isin(linhas_usadas)
        ]

        candidatas.append(df)

    if not candidatas:
        print(f"{arquivo}: nenhuma linha candidata")
        resumo.append({"arquivo": arquivo, "baixadas": len(ja_ok), "tentadas": 0, "candidatas": 0})
        continue

    todos = pd.concat(candidatas, ignore_index=True)
    todos = todos.sample(frac=1, random_state=42).reset_index(drop=True)

    registros_especie = ja_ok.to_dict("records")
    baixadas = len(ja_ok)
    tentadas = 0
    lote = []

    for _, linha in todos.iterrows():
        if baixadas >= N_POR_ESPECIE:
            break

        tentadas += 1

        aba = linha["__aba"]
        linha_excel = int(linha["__linha_excel"])
        id_item = str(linha["__id_item"]).strip()
        original_url = str(linha["__url"]).strip()
        licenca = linha.get("license", "")

        if aba == "GBIF" and (pd.isna(licenca) or str(licenca).strip() == ""):
            licenca = consultar_licenca_gbif(id_item)

        if not licenca_permitida(licenca):
            registro = {
                "arquivo": arquivo,
                "aba": aba,
                "linha_excel": linha_excel,
                "id_item": id_item,
                "filename": "",
                "original_url": original_url,
                "download_url": "",
                "licenca": licenca,
                "status": "ignorada_licenca",
                "fallback_usado": False,
            }
            controle = pd.concat([controle, pd.DataFrame([registro])], ignore_index=True)
            continue

        url_download = normalizar_url(original_url)
        ext = extensao_da_url(url_download)

        tmp_filename = f"tmp_{id_item}_{linha_excel}{ext}".replace("/", "_")
        tmp_path = pasta_especie / tmp_filename

        lote.append({
            "arquivo": arquivo,
            "aba": aba,
            "linha_excel": linha_excel,
            "id_item": id_item,
            "original_url": original_url,
            "url_download": url_download,
            "licenca": licenca,
            "ext": ext,
            "tmp_path": tmp_path,
        })

        if len(lote) >= TAMANHO_LOTE:
            controle, registros_especie, baixadas = processar_lote(
                lote, controle, registros_especie, pasta_especie, especie, arquivo, baixadas
            )

            controle.to_csv(ARQUIVO_CONTROLE, index=False)
            lote = []

    if lote and baixadas < N_POR_ESPECIE:
        controle, registros_especie, baixadas = processar_lote(
            lote, controle, registros_especie, pasta_especie, especie, arquivo, baixadas
        )

        controle.to_csv(ARQUIVO_CONTROLE, index=False)

    resumo.append({
        "arquivo": arquivo,
        "baixadas": baixadas,
        "tentadas": tentadas,
        "candidatas": len(todos),
    })

    print(
        f"{arquivo}: finalizado | "
        f"baixadas={baixadas} | tentadas={tentadas} | candidatas={len(todos)}"
    )


resumo_df = pd.DataFrame(resumo)

print("\nResumo final:")
print(resumo_df.to_string(index=False))

print("\nTotal baixado:")
print(resumo_df["baixadas"].sum())

print("\nFinalizado.")
print(f"Imagens salvas em: {PASTA_IMAGENS}")
print(f"Controle salvo em: {ARQUIVO_CONTROLE}")
