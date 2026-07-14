from fpdf import FPDF

pdf = FPDF()
pdf.add_page()

# Title
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Acme Corp Financial Report 2026", ln=True, align="C")
pdf.ln(10)

# Body
pdf.set_font("Arial", "", 12)
text = """Company Name: Acme Corp
Year: 2026

Financial Overview:
Acme Corp had a very successful year in 2026. The total revenue reached $50 million, and the net profit was $12 million. 

Key Highlights:
- The company launched a new AI software called AcmeBot.
- AcmeBot alone contributed to 30% of the total revenue.
- CEO Jane Doe stated that she expects a 20% growth next year due to their expansion in the European market.

Risks:
- The supply chain issues in Asia caused a 5% delay in hardware manufacturing.
- The company plans to open a new factory in Mexico to mitigate this risk."""

pdf.multi_cell(0, 8, text)

pdf.output("sample_financial_report.pdf")
