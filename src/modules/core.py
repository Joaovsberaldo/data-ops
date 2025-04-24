import os
import json
from typing import Any, Dict, List
from itertools import zip_longest
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from markdown2 import markdown
from weasyprint import HTML
from dotenv import load_dotenv
load_dotenv("/.env")
from modules.utils import(
    read_json_lines,
    load_products,
    clean_json_data,
    add_feedback_ids,
    write_json,
    read_json_file,
    ensure_directory,
    list_image_files,
    encode_image,
    write_text_file,
    extract_field_from_json,
)

class FeedbackLoader: 
    def __init__(self, feedback_path: str, product_path: str):
        self.feedback_path = feedback_path
        self.product_path = product_path        
        
    def extract_feedbacks(self, output_path: str, limit: int) -> List[Dict[str, Any]]:
        """Executa as funções que extraem os dados de feedback e salva em um arquivo JSON"""
        # -------------------- Carrega dados --------------------    
        raw_feedbacks = read_json_lines(self.feedback_path, limit)
        products = load_products(self.product_path)
        
        # -------------------- Processa dados --------------------
        processed_feedbacks = [
            {
                "rating": fb.get("rating"),
                "title": fb.get("title"),
                "text": fb.get("text"),
                "product_name": products.get(fb.get("parent_asin"), "Desconhecido")
            }
            for fb in raw_feedbacks
        ]
        
        cleaned = [clean_json_data(item) for item in processed_feedbacks]
        cleaned = [item for item in cleaned if item]
        cleaned = add_feedback_ids(cleaned)
        
        # -------------------- Salva o arquivo --------------------
        write_json(cleaned, output_path)
        return cleaned
    
class FeedbackSummarizer():
    def __init__(self, input_path: str, client = OpenAI()):
        self.input_path = input_path
        self.client = client
        
    def generate_summary(
        self,
        prompt: str,
        example: str,
        model: str,
        output_path: str, 
    ) -> List[Dict[str, Any]]:
        """Executa o fluxo completo: leitura de feedbacks, geração de resumos e salvamento final dos dados."""

        items = read_json_file(self.input_path)
        
        # Extrai campos para prompt
        serialized = [
            json.dumps({
                "title": it.get("title"),
                "text": it.get("text"),
                "product_name": it.get("product_name")
            }, ensure_ascii=False)
            for it in items
        ]

        # Monta e envia prompt
        prompt = prompt.format(
            lista_feedback_cliente="\n".join(serialized),
            example=example
        )
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": prompt}],
            text={
            "format": {
                "type": "json_schema",
                "name": "feedback_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summaries": {
                            "description": "Lista de resumos dos feedbacks dos clientes",
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "summary": {
                                        "type": "string",
                                        "description": "Resumo do feedback do cliente"
                                    },
                                    "situation": {
                                        "description": "Lista de situações extraídas do feedback",
                                        "type": "array",
                                        "items": {
                                        "type": "object",
                                        "properties": {
                                            "keyword": {
                                            "type": "string",
                                            "description": "Palavra-chave relacionada à situação"
                                            },
                                            "categories": {
                                                "type": "array",
                                                "items": { "type": "string" },
                                                "description": "Lista de categorias associadas à palavra-chave"
                                            }
                                        },
                                        "required": ["keyword", "categories"],
                                        "additionalProperties": False
                                        },
                                    }
                                },
                                "required": ["summary", "situation"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["summaries"],
                    "additionalProperties": False
                }
            }
        },
            stream=True
        )
        
        result_json = ""
        # Itera sobre os eventos e acumula o texto quando o evento 'response.completed' for recebido
        for event in response:
            if event.type == 'response.refusal.delta':
                print(event.delta, end="")

            # Exibe resposta em streaming    
            elif event.type == 'response.output_text.delta':
                print(event.delta, end="", flush=True,)

            elif event.type == 'response.error':
                print(event.error, end="")

            # Cria arquivo JSON com o conteúdo da resposta
            elif event.type == 'response.completed':
                # Define response_completed e encerra o loop se necessário
                result_json = event.response.output[0].content[0].text        
        result = json.loads(result_json)
        
        # Tratamento
        summaries = result.get("summaries", [])
        final = []
        for s, fb in zip_longest(summaries, items):
            if s is None:
                print("⚠️ Aviso: feedback extra sem resumo correspondente.")
                continue
            if fb is None:
                print("⚠️ Aviso: resumo extra sem feedback correspondente.")
                continue
    
            final.append({
                "feedback_id": fb.get("feedback_id"),
                "rating": fb.get("rating"),
                **s
            })

        # Persiste
        write_json(final, output_path)
        return final       
    
class GraphicGenerator:
    """Gera gráficos de distribuição de ratings e características."""
    
    def __init__(self, input: List[Dict[str, Any]], output_dir: str):
        self.input = input
        self.output_dir = output_dir
        ensure_directory(output_dir)

    def _get_ratings(
        self, 
        key: str = "rating"
        ) -> List[int]:
        return [item.get(key) for item in self.input]
    
    def _get_values(
        self, 
        key: str
        ) -> List[List[str]]:
        return [
            [
                value for situation in item.get("situation", []) 
                for value in (situation.get(key) if isinstance(situation.get(key), list) else [situation.get(key)])
            ]
            for item in self.input
        ]

    def _prepare_data(
        values: List[List[str]], 
        ratings: List[int], 
        label: str
        ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {label: v, "Nota": r}
                for vals, r in zip(values, ratings)
                for v in vals
            ]
        )

    def plot_rating_distribution(
        self, 
        ) -> None:
        """Plota e salva histograma de notas."""
        ratings = self._get_ratings()
        plt.figure(figsize=(8, 5))
        sns.histplot(ratings, bins=range(1, 7), kde=False, color='skyblue', edgecolor='black')
        plt.xlabel('Nota')
        plt.ylabel('Frequência')
        plt.title('Distribuição das Notas')
        plt.grid(axis='y', linestyle='--')
        plt.xticks(range(1, 6))
        plt.savefig(os.path.join(self.output_dir, "distribuicao_notas.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_categorical_distribution(
        self,
        key: str,
        label: str,  
        filename: str,
        ) -> None:
        """Plota e salva gráfico de distribuição de notas por categoria ou keyword."""
        values = self._get_values(key)
        ratings = self._get_ratings()
        df = GraphicGenerator._prepare_data(values, ratings, label)
        if df.empty:
            print(f"Aviso: Sem dados para '{label}'.")
            return
        
        pivot = df.pivot_table(index=label, columns='Nota', aggfunc='size', fill_value=0)
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values(by="Total", ascending=False).drop(columns="Total")
        
        pivot.plot(kind='barh', stacked=True, figsize=(8, 5), colormap='coolwarm')
        plt.gca().invert_yaxis()
        plt.xlabel('Frequência')
        plt.ylabel(label)
        plt.title(f"Distribuição de notas por {label}")
        plt.legend(title='Nota', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='x', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
        
    def generate_all(self) -> List[str]:
        """Gera todos os gráficos e retorna nomes de arquivos gerados."""
        self.plot_rating_distribution()
        self.plot_categorical_distribution('keyword', 'Keyword', 'keyword_distribution.png')
        self.plot_categorical_distribution('categories', 'Category', 'category_distribution.png')
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        print('Arquivos gerados:', files)
        return files
    
class GraphicAnalyzer:
    """Analisa imagens via OpenAI e salva a resposta."""
    
    def __init__(self, input_dir: str, output_path: str, client = OpenAI()):
        self.input_dir = input_dir
        self.output_path = output_path
        self.client = client
    
    def analyze(
        self,
        model: str, 
        prompt: str, 
        system_prompt: str, 
        ) -> Any:
        """Codifica imagens, envia ao modelo e salva a análise."""
        images_paths = list_image_files(self.input_dir)
        images_b64 = encode_image(images_paths)
        
        input_images = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{img}", "detail": "low"}
            for img in images_b64
        ]
          
        response = self.client.responses.create(
            model=model, 
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}] + input_images}
            ],
            stream=True
        )
        
        # Imprime resposta em streaming
        content = ""
        for event in response:
            if event.type == 'response.refusal.delta':
                print(event.delta, end="")
            elif event.type == 'response.output_text.delta':
                print(event.delta, end="", flush=True)
            elif event.type == 'response.error':
                print(event.error, end="")
                
            # Gera arquivo da resposta final
            elif event.type == 'response.completed':
                content = event.response.output[0].content[0].text
                write_text_file(self.output_path, content)
        return content
        
class SummaryAnalyzer:
    """Executa análise textual sobre os resumos de feedback."""
    
    def __init__(self, input_path: str, output_path: str, client = OpenAI()):
        self.input_path = input_path
        self.output_path = output_path
        self.client = client    
    
    def analyze(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: str, 
        ) -> Any:
        """Gera análise textual via modelo da OpenAI."""
        
        # Extrai os contexto
        summaries = extract_field_from_json(self.input_path, "summary")
        
        # Estrutura prompt
        prompt = prompt.format(summaries="\n".join(summaries))
        
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
    
        # Imprime a resposta em streaming
        content = ""
        for event in response:
            if event.type == 'response.refusal.delta':
                print(event.delta, end="")
            elif event.type == 'response.output_text.delta':
                print(event.delta, end="", flush=True)
            elif event.type == 'response.error':
                print(event.error, end="")
                
            # Gera arquivo da resposta final
            elif event.type == 'response.completed':
                content = event.response.output[0].content[0].text
                write_text_file(self.output_path, content)
        return content

class DataAnalyzer:
    """Executa análise final com base nas análises de resumos e gráficos."""
    
    def __init__(self, summaries_analysis: str, graphics_analysis: str, output_path: str, client = OpenAI()):
        self.summaries_analysis = summaries_analysis
        self.graphics_analysis = graphics_analysis
        self.output_path = output_path
        self.client = client
    
    def analyze(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: str, 
        ) -> Any:
        """Gera análise executiva final com base nas análises prévias."""
        
        # Estrutura o prompt
        prompt = prompt.format(
            analise_graficos=self.graphics_analysis, 
            analise_resumos=self.summaries_analysis
        )
        
        # Envia requisição para modelo
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        # Imprime resposta em streaming
        content = ""
        for event in response:
            if event.type == 'response.refusal.delta':
                print(event.delta, end="")
            elif event.type == 'response.output_text.delta':
                print(event.delta, end="", flush=True)
            elif event.type == 'response.error':
                print(event.error, end="")
                
            # Gera arquivo da resposta final
            elif event.type == 'response.completed':
                content = event.response.output[0].content[0].text
                write_text_file(self.output_path, content)
        return content

class ReportGenerator:
    def __init__(self, input_path: str, graphics_dir: str, output_path: str, client = OpenAI()) -> Any:
        self.input_path = input_path
        self.graphics_dir = graphics_dir
        self.output_path = output_path
        self.client = client
        
    def generate_report(
        self,
        model: str, 
        prompt: str, 
        system_prompt: str, 
        ) -> Any:
        """Gera análise textual via modelo da OpenAI."""
        
        # Estruturar input imagens
        images_paths = list_image_files(self.graphics_dir)    
        images_b64 = encode_image(images_paths)        
        input_images = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{img}", "detail": "low"}
            for img in images_b64
        ]
        
        # Estruturar prompt
        prompt = prompt.format(
            analise_dados=self.input_path, 
            nome_graficos=images_paths
        )

        # Enviar requisição para modelo
        response = self.client.responses.create(
            model=model,
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}] + input_images}
            ],
            stream=True
        )

        # Imprime resposta em streaming
        content = ""
        for event in response:
            if event.type == 'response.refusal.delta':
                print(event.delta, end="")
            elif event.type == 'response.output_text.delta':
                print(event.delta, end="", flush=True)
            elif event.type == 'response.error':
                print(event.error, end="")
                
            # Gera arquivo da resposta final
            elif event.type == 'response.completed':
                content = event.response.output[0].content[0].text
                self._write_pdf(content)
        return content

    def _write_pdf(self, content: str) -> None:
        """Converte markdown para HTML e gera um PDF estilizado."""
        html = markdown(content, extras=["tables"])
        styled_html = self._wrap_with_css(html)
        HTML(string=styled_html, base_url="data_ops/").write_pdf(self.output_path)
        print(f"Relatório salvo com sucesso no arquivo: {self.output_path}")

    def _wrap_with_css(self, html: str) -> str:
        """Aplica estilos CSS ao HTML gerado para o PDF."""
        css = """
        <style>
            img {
                max-width: 400px;
                height: auto;
                display: block;
                margin: 2em auto;
                page-break-inside: avoid;
                break-inside: avoid;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 1em;
                break-inside: avoid;
                page-break-inside: avoid;
            }
            tr {
                break-inside: avoid;
                page-break-inside: avoid;
            }
            th, td {
                border: 1px solid #333;
                padding: 8px;
                text-align: left;
            }
            th {
                font-weight: bold;
                background-color: #eee;
            }
        </style>
        """
        return css + html
