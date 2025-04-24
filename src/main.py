from modules.core import(
    FeedbackLoader,
    FeedbackSummarizer,
    GraphicGenerator,
    GraphicAnalyzer,
    SummaryAnalyzer,
    DataAnalyzer,
    ReportGenerator,
    ensure_directory,
)
from modules.utils import read_text_file
import os
from dotenv import load_dotenv
load_dotenv("/.env")

# -------------------- Constantes de Caminhos --------------------
DATA_DIR = "../data"
OUTPUT_DIR = "output"
INPUT_DIR = "./output"
GRAPHICS_DIR = os.path.join(INPUT_DIR, "graphics")
PROMPT_DIR = "./prompts/user"
SYSTEM_PROMPT_DIR = "./prompts/system"
RESPONSE_EXAMPLE_DIR = "./prompts/examples"

def main(limit: int):
    ensure_directory(OUTPUT_DIR)
    ensure_directory(GRAPHICS_DIR)

    # -------------------- 1. Carregar Feedbacks --------------------
    feedback_loader = FeedbackLoader(
        feedback_path=os.path.join(DATA_DIR, "Subscription_Boxes.jsonl"),
        product_path=os.path.join(DATA_DIR, "meta_Subscription_Boxes.jsonl")
    )
    feedback_loader.extract_feedbacks(
        output_path=os.path.join(OUTPUT_DIR, "feedbacks.json"),
        limit=limit
    )

    # -------------------- 2. Resumir Feedbacks --------------------
    summarizer = FeedbackSummarizer(input_path=os.path.join(INPUT_DIR, "feedbacks.json"))
    
    summary_prompt = read_text_file(os.path.join(PROMPT_DIR, "summary.txt"))
    summary_example = read_text_file(os.path.join(RESPONSE_EXAMPLE_DIR, "summary.txt"))
    
    summaries = summarizer.generate_summary(
        prompt=summary_prompt,
        response_example=summary_example,
        model="gpt-4.1-mini",
        output_path=os.path.join(OUTPUT_DIR, "summary.json")
    )

    # -------------------- 3. Gerar Gráficos --------------------
    graphic_generator = GraphicGenerator(input=summaries, output_dir=GRAPHICS_DIR)
    graphic_generator.generate_all()

    # -------------------- 4. Analisar Gráficos --------------------
    graphic_analyzer = GraphicAnalyzer(
        input_dir=GRAPHICS_DIR,
        output_path=os.path.join(OUTPUT_DIR, "graphic_analyze.md")
    )
    graphics_analysis = graphic_analyzer.analyze(
        model="gpt-4.1-mini",
        prompt=read_text_file(os.path.join(PROMPT_DIR, "analyze.txt")),
        system_prompt=read_text_file(os.path.join(SYSTEM_PROMPT_DIR, "analyze_graphic.txt"))
    )

    # -------------------- 5. Analisar Resumos --------------------
    summary_analyzer = SummaryAnalyzer(
        input_path=os.path.join(INPUT_DIR, "summary.json"),
        output_path=os.path.join(OUTPUT_DIR, "summary_analyze.md")
    )
    summaries_analysis = summary_analyzer.analyze(
        model="gpt-4.1-nano",
        prompt=read_text_file(os.path.join(PROMPT_DIR, "analyze_summary.txt")),
        system_prompt=read_text_file(os.path.join(SYSTEM_PROMPT_DIR, "analyze_summary.txt"))
    )

    # -------------------- 6. Análise Executiva --------------------
    data_analyzer = DataAnalyzer(
        summaries_analysis=summaries_analysis,
        graphics_analysis=graphics_analysis,
        output_path=os.path.join(OUTPUT_DIR, "data_analyze.md")
    )
    data_analysis = data_analyzer.analyze(
        model="gpt-4.1-mini",
        prompt=read_text_file(os.path.join(PROMPT_DIR, "final_analyze.txt")),
        system_prompt=read_text_file(os.path.join(SYSTEM_PROMPT_DIR, "final_analyze.txt"))
    )

    # -------------------- 7. Geração do Relatório Final --------------------
    report_generator = ReportGenerator(
        input_path=data_analysis,
        graphics_dir=GRAPHICS_DIR,
        output_path=os.path.join(OUTPUT_DIR, "executive_report.pdf")
    )
    report_generator.generate_report(
        model="gpt-4.1",
        prompt=read_text_file(os.path.join(PROMPT_DIR, "report.txt")),
        system_prompt=read_text_file(os.path.join(SYSTEM_PROMPT_DIR, "report.txt"))
    )

if __name__ == "__main__":
    main(10)