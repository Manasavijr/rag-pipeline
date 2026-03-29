"""
Sample document generator — creates realistic test documents for the RAG pipeline.
Generates PDFs, TXT files covering legal, financial, and technical content.

Run: python data/generate_sample_docs.py
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Sample document content ───────────────────────────────────────────────────

LEGAL_CONTRACT = """
SERVICES AGREEMENT

This Services Agreement ("Agreement") is entered into as of January 1, 2025, between
TechCorp Inc. ("Client") and DataSolutions LLC ("Provider").

1. SCOPE OF SERVICES
Provider agrees to deliver the following services:
- Machine learning model development and deployment
- Data pipeline architecture and implementation
- Technical documentation and knowledge transfer
- Monthly performance reporting and optimization

2. PAYMENT TERMS
Client agrees to pay Provider a monthly retainer of $15,000, due within 30 days
of invoice. Late payments incur a 1.5% monthly interest charge. Annual contract
value is $180,000 with a 10% discount for upfront payment.

3. CONFIDENTIALITY
Both parties agree to maintain strict confidentiality of all proprietary information,
trade secrets, and business data shared during the engagement. This obligation
survives termination of the Agreement for a period of five (5) years.

4. INTELLECTUAL PROPERTY
All deliverables created under this Agreement shall be considered work-for-hire.
Client retains full ownership of all developed models, code, and documentation.
Provider retains ownership of pre-existing tools and frameworks.

5. TERMINATION
Either party may terminate this Agreement with 30 days written notice.
Client may terminate immediately for cause if Provider materially breaches
any term of this Agreement.

6. LIMITATION OF LIABILITY
Provider's total liability shall not exceed the fees paid in the preceding
three months. Neither party shall be liable for indirect or consequential damages.

7. GOVERNING LAW
This Agreement shall be governed by the laws of California, USA.
"""

FINANCIAL_REPORT = """
QUARTERLY FINANCIAL REPORT — Q3 2025
TechVentures Capital Fund

EXECUTIVE SUMMARY
The fund delivered strong performance in Q3 2025 with total returns of 18.3%,
outperforming the benchmark S&P 500 index by 6.2 percentage points.

PORTFOLIO PERFORMANCE
Total AUM: $2.4 billion
Net Returns: 18.3% YTD
Sharpe Ratio: 2.1
Maximum Drawdown: -4.2%

TOP PERFORMERS
1. AI Infrastructure Holdings — +42% (largest position at 8.3% of portfolio)
2. Cloud Computing ETF — +28% (defensive position, 5.1% allocation)
3. Semiconductor Index Fund — +31% (cyclical play, 4.7% allocation)

RISK METRICS
Portfolio Beta: 1.15
VaR (95%, 1-day): $24.3 million
Correlation to S&P 500: 0.73

SECTOR ALLOCATION
Technology: 42%
Healthcare: 18%
Financial Services: 15%
Energy: 10%
Consumer Staples: 8%
Other: 7%

MARKET OUTLOOK
We maintain a cautiously optimistic outlook for Q4 2025. Key risks include:
- Federal Reserve interest rate decisions
- Geopolitical tensions in key markets
- AI regulatory developments in EU and US
- Supply chain disruptions in semiconductor sector

The fund plans to increase exposure to AI infrastructure by 3% in Q4,
reducing allocation to traditional financial services.

FEES AND EXPENSES
Management Fee: 1.5% annually
Performance Fee: 20% above 8% hurdle rate
Total Expense Ratio: 1.8%
"""

TECHNICAL_DOC = """
SYSTEM ARCHITECTURE DOCUMENT
MLOps Platform v2.0 — Technical Reference

1. OVERVIEW
The MLOps platform provides end-to-end machine learning lifecycle management,
from data ingestion through model deployment and monitoring.

2. ARCHITECTURE COMPONENTS

2.1 Data Ingestion Layer
- Apache Kafka for real-time streaming data (throughput: 1M events/sec)
- Apache Spark for batch processing (PySpark 3.5)
- Data quality validation using Great Expectations
- Schema registry with Confluent Schema Registry

2.2 Model Training Infrastructure
- Kubernetes cluster (GKE) with 50 GPU nodes (NVIDIA A100)
- MLflow for experiment tracking and model registry
- Ray for distributed training across multiple nodes
- Automated hyperparameter tuning with Optuna

2.3 Model Serving Layer
- FastAPI inference servers (uvicorn + gunicorn)
- ONNX Runtime for optimized inference (3x speedup vs PyTorch)
- Redis caching for frequent predictions (99th percentile: 12ms)
- Load balancing with NGINX and auto-scaling via HPA

2.4 Monitoring and Observability
- Prometheus + Grafana for metrics
- Jaeger for distributed tracing
- ELK Stack for centralized logging
- Custom drift detection using KS-test (p-value threshold: 0.05)

3. PERFORMANCE BENCHMARKS
Model Inference Latency: p50=8ms, p95=45ms, p99=120ms
Training Throughput: 1,000 examples/second on A100
Data Processing: 500GB/hour batch, 50MB/second streaming
Uptime SLA: 99.9% (8.7 hours downtime/year maximum)

4. SECURITY
- mTLS for all service-to-service communication
- RBAC with OPA (Open Policy Agent)
- Secrets management via HashiCorp Vault
- SOC 2 Type II compliant infrastructure

5. DEPLOYMENT
Rolling deployments with zero-downtime
Blue-green deployment for major releases
Canary releases at 1%, 10%, 50%, 100%
Automated rollback on error rate threshold (>1%)
"""

AI_RESEARCH = """
RESEARCH PAPER SUMMARY: Advances in Large Language Model Fine-tuning

ABSTRACT
This paper presents a comprehensive analysis of fine-tuning techniques for large
language models (LLMs), comparing LoRA, QLoRA, full fine-tuning, and instruction
tuning approaches across multiple downstream tasks.

KEY FINDINGS

1. Parameter-Efficient Fine-tuning (PEFT)
LoRA achieves 94% of full fine-tuning performance while updating only 0.1% of
parameters. This reduces GPU memory requirements by 8x, enabling fine-tuning
of 70B parameter models on consumer hardware.

2. Data Quality vs. Quantity
Experiments show that 1,000 high-quality curated examples outperform 100,000
noisily-labeled examples by an average of 12% across all benchmark tasks.
Data curation using LLM-based filtering showed the strongest results.

3. Instruction Tuning Results
Models fine-tuned with chain-of-thought instruction data showed 23% improvement
on reasoning benchmarks (MMLU, HellaSwag) compared to standard instruction tuning.
The optimal instruction dataset size was found to be 50,000-100,000 examples.

4. Quantization Impact
4-bit quantization (QLoRA) reduces model size by 4x with only 2.3% performance
degradation on average. 8-bit quantization shows <1% performance loss and is
recommended for production deployments.

BENCHMARK RESULTS
Task                | Full FT | LoRA  | QLoRA | Baseline
MMLU                | 78.4%   | 76.2% | 74.8% | 70.1%
HumanEval (coding)  | 52.3%   | 50.1% | 48.7% | 38.2%
GSM8K (math)        | 71.2%   | 69.8% | 67.3% | 55.4%
TruthfulQA          | 64.1%   | 62.3% | 61.0% | 52.7%

CONCLUSION
LoRA-based fine-tuning represents the optimal balance of performance and
computational efficiency for most practical applications. We recommend QLoRA
for resource-constrained environments and full fine-tuning only when maximum
performance is critical and compute budget is unlimited.
"""


def write_txt(filename: str, content: str) -> None:
    path = RAW_DIR / filename
    path.write_text(content.strip(), encoding="utf-8")
    logger.info(f"Created: {path} ({len(content)} chars)")


def create_pdf(filename: str, content: str) -> None:
    """Create PDF using reportlab if available, else TXT fallback."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch

        path = RAW_DIR / filename
        doc = SimpleDocTemplate(str(path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        for line in content.strip().split("\n"):
            if line.strip():
                if line.isupper() and len(line) < 60:
                    story.append(Paragraph(line, styles["Heading1"]))
                else:
                    story.append(Paragraph(line, styles["Normal"]))
            else:
                story.append(Spacer(1, 0.1 * inch))

        doc.build(story)
        logger.info(f"Created PDF: {path}")
    except ImportError:
        # Fallback to TXT if reportlab not installed
        txt_path = RAW_DIR / filename.replace(".pdf", ".txt")
        txt_path.write_text(content.strip(), encoding="utf-8")
        logger.info(f"reportlab not installed — created TXT fallback: {txt_path}")


def main():
    logger.info("Generating sample documents...")

    write_txt("legal_contract.txt", LEGAL_CONTRACT)
    write_txt("financial_report.txt", FINANCIAL_REPORT)
    write_txt("technical_architecture.txt", TECHNICAL_DOC)
    write_txt("ai_research_summary.txt", AI_RESEARCH)

    create_pdf("legal_contract.pdf", LEGAL_CONTRACT)
    create_pdf("financial_report.pdf", FINANCIAL_REPORT)
    create_pdf("technical_architecture.pdf", TECHNICAL_DOC)
    create_pdf("ai_research_summary.pdf", AI_RESEARCH)

    logger.info(f"Done! Files created in {RAW_DIR}/")
    logger.info("Upload them via: POST /api/v1/ingest/upload")


if __name__ == "__main__":
    main()
