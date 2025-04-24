# 📊 DataOps – Agents for Analyzing Amazon Customer Feedback

**DataOps** is a set of agents developed to **automate the analysis of Amazon customer feedback**. The project uses real-world data in `.jsonl` format and focuses on turning raw text into **actionable insights** for data, product, marketing, and customer experience teams.

---

## ✅ Problems We Solve

The project automates several stages of the feedback analysis workflow:

- 🧹 Data extraction and cleaning  
- 📖 Text interpretation using language models  
- 📊 Chart generation  
- 📈 Automated data analysis  
- 📝 Executive report creation in PDF format  

**Key benefits:**  
- Saves time and manual effort  
- Supports decision-making  
- Provides quick access to strategic insights  

---

## 🚀 How to Install and Run the Project

### ✅ Requirements

- Python 3.8 or higher  
- `virtualenv` (or equivalent)

### 🧪 Step-by-step

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/data_ops.git
cd data_ops
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv           # `.venv` para ocultar a pasta
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**

Create a .env file in the project root and add your custom settings.

---

### ▶️ How to Use
1. **Set the number of feedback entries to analyze by adjusting the limit parameter in the code:**

```python
limit = 10
```

2. **Run the project:**
```bash
python src/main.py
```

📥 **Input:**
A .jsonl file with Amazon customer feedback (located in data_ops/data/)

📤 **Saída:**
A PDF report saved to output/executive_report.pdf

---

## 🧱 Project Structure
<pre> 
📦 data_ops/
┣ 📂 data/              → Input data (.jsonl)
┣ 📂 output/            → Generated results (charts, reports, etc.)
┣ 📂 src/               → Main source code
┃ ┣ 📂 modules/         → Reusable functions (core, utils, etc.)
┃ ┣ 📂 prompts/         → Agent prompts (user, system, examples)
┃ ┗ 📜 main.py          → Main script
┣ 📜 requirements.txt   → Project dependencies
┗ 📜 README.md          → Project documentation
</pre>

---

## 📌 Next Steps
🔍 Data Analyst Agent:
Add an agent that queries a database directly for analysis.
This improvement is essential for enterprise environments where data is stored in relational systems.
