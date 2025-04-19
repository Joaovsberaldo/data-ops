import os
import json
import uuid
import base64
from typing import List, Dict, Any, Optional
from itertools import islice

# -------------------- I/O UTILS --------------------

def ensure_directory(path: str) -> None:
    """Garante que o diretório existe."""
    os.makedirs(path, exist_ok=True)


def read_json_file(path: str) -> List[Dict[str, Any]]:
    """Lê um arquivo JSON completo."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_json_lines(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Lê linhas JSON de um arquivo, com limite opcional."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = islice(f, limit) if limit else f
        return [json.loads(line.strip()) for line in lines]


def read_text_file(file_path: str) -> str:
    """Lê arquivos de textos"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_json(data: Any, path: str) -> None:
    """Salva dados em formato JSON com identação."""
    ensure_directory(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text_file(path: str, content: str) -> None:
    """Salva um conteúdo textual em arquivo."""
    ensure_directory(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


# -------------------- FEEDBACK UTILS --------------------

def clean_json_data(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Limpa um item JSON escapando campos e corrigindo estrutura."""
    try:
        text = item.get("text", "")
        item["text"] = json.loads(json.dumps(text))  # escapa aspas corretamente
        json_str = json.dumps(item, ensure_ascii=False)
        json_str = json_str.replace(", }", " }" ).replace(", ]", " ]")
        return json.loads(json_str)
    except (TypeError, json.JSONDecodeError) as e:
        print(f"Erro ao limpar JSON: {e}")
        return None


def add_feedback_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adiciona um ID único a cada item de feedback."""
    feedbacks = []
    for item in items:
        item["feedback_id"] = str(uuid.uuid4())
        feedbacks.append(item)
    return feedbacks


def load_products(file_path: str) -> Dict[str, str]:
    """Carrega metadados de produtos e retorna dict parent_asin → title."""
    products = {}
    for item in read_json_lines(file_path):
        pid = item.get("parent_asin")
        title = item.get("title", "Desconhecido")
        if pid:
            products[pid] = title
    return products


# -------------------- IMAGEM UTILS --------------------

def list_image_files(directory: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.svg', '.webp')) -> List[str]:
    """Lista caminhos de arquivos de imagem em um diretório."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ]


def encode_image(img_paths: List[str]) -> List[str]:
    """Codifica imagens como strings base64."""
    encoded = []
    for p in img_paths:
        with open(p, 'rb') as img:
            encoded.append(base64.b64encode(img.read()).decode('utf-8'))
    return encoded


def extract_field_from_json(path: str, field: str) -> List[str]:
    """Extrai um campo específico de todos os registros em um JSON."""
    data = read_json_file(path)
    return [item.get(field, "") for item in data]