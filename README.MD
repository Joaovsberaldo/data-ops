# 📊 DataOps – Agentes para Análise de Feedbacks da Amazon

**DataOps** é um conjunto de agentes desenvolvidos para **automatizar a análise de feedbacks de clientes da Amazon**. O projeto utiliza dados reais em formato `.jsonl` e tem como foco transformar texto bruto em **insights acionáveis** para equipes de dados, produto, marketing e experiência do cliente.

---

## ✅ Problemas que resolvemos

O projeto automatiza diversas etapas do fluxo de análise:

- 🧹 Extração e tratamento de dados  
- 📖 Interpretação de textos com modelos de linguagem  
- 📊 Geração de gráficos  
- 📈 Análise automatizada dos dados  
- 📝 Criação de relatórios executivos em PDF  

**Benefícios principais:**  
- Economia de tempo e esforço manual  
- Suporte à tomada de decisão  
- Acesso rápido a insights estratégicos  

---

## 🚀 Como instalar e rodar o projeto

### ✅ Pré-requisitos

- Python 3.8 ou superior  
- `virtualenv` (ou similar)

### 🧪 Passo a passo

1. **Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/data_ops.git
cd data_ops
```

2. **Crie e ative o ambiente virtual:**
```bash
python -m venv venv           # `.venv` para ocultar a pasta
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

3. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente:**

Crie um arquivo `.env` na raiz do projeto com suas variáveis.

---

### ▶️ Como usar
1. **Defina o número de feedbacks que deseja analisar ajustando o parâmetro limit no código:**

```python
limit = 10
```

2. **Execute o projeto:**
```bash
python src/main.py
```

📥 *Entrada:*
Arquivo .jsonl com feedbacks da Amazon (localizado em `data_ops/data/`)

📤 *Saída:*
Relatório em PDF salvo em `output/executive_report.pdf`

---

## 🧱 Estrutura do projeto
<pre> 
📦 data_ops/
┣ 📂 data/              → Dados de entrada (.jsonl)
┣ 📂 output/            → Resultados gerados (gráficos, relatórios etc.)
┣ 📂 src/               → Código-fonte principal
┃ ┣ 📂 modules/         → Funções reutilizáveis (core, utils, etc.)
┃ ┣ 📂 prompts/         → Prompts para agentes (usuário, sistema, exemplos)
┃ ┗ 📜 main.py          → Script principal
┣ 📜 requirements.txt   → Dependências
┗ 📜 README.md          → Documentação do projeto
</pre>

---

## 📌 Próximos passos
🔍 Agente Analista de Dados:
Adicionar um agente que consulta diretamente um banco de dados para análises.
Essa melhoria é essencial para ambientes corporativos com dados armazenados em sistemas relacionais.
