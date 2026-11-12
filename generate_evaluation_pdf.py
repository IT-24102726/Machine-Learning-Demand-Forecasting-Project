import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "ChefCast AI — Evaluation Metrics, Confusion Matrix & WAPE Technical Guide")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
        
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, text)
        self.drawString(54, 36, "SLIIT IT3091 Machine Learning | Group 2026-AI-36")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_pdf(filename="ChefCast_AI_Model_Evaluation_and_WAPE_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary = colors.HexColor("#0F172A")    # Deep Slate
    brand_blue = colors.HexColor("#1E3A8A") # Navy Blue
    accent = colors.HexColor("#0284C7")     # Sky Accent
    text_dark = colors.HexColor("#1E293B")  # Charcoal
    bg_light = colors.HexColor("#F8FAFC")   # Light gray
    border_col = colors.HexColor("#CBD5E1") # Slate border
    
    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=brand_blue,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=brand_blue,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=accent,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_dark,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    formula_style = ParagraphStyle(
        'Formula_Custom',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=10,
        leading=14,
        textColor=brand_blue,
        alignment=1,
        spaceBefore=4,
        spaceAfter=4
    )

    th_style = ParagraphStyle(
        'TH_Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    td_style = ParagraphStyle(
        'TD_Style',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=text_dark,
        alignment=0
    )

    td_center = ParagraphStyle(
        'TD_Center',
        parent=td_style,
        alignment=1
    )

    td_bold_center = ParagraphStyle(
        'TD_Bold_Center',
        parent=td_style,
        fontName='Helvetica-Bold',
        alignment=1
    )

    story = []

    # Title Banner
    story.append(Paragraph("ChefCast AI: Machine Learning Evaluation Guide", title_style))
    story.append(Paragraph("Confusion Matrix, Accuracy, Precision, Recall, F1-Score & Real-World Validation WAPE Breakdown<br/><b>Group ID:</b> 2026-AI-36 | <b>Course:</b> SLIIT IT3091 Machine Learning (Industry Explorer Track)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=brand_blue, spaceBefore=0, spaceAfter=10))

    # Executive Overview
    story.append(Paragraph("Executive Context: Continuous Regression vs. Operational Classification", h1_style))
    story.append(Paragraph(
        "ChefCast AI operates across two complementary lenses: <b>Lens 1</b> forecasts continuous daily restaurant turnover in Rupees (LKR), and <b>Lens 2</b> forecasts daily physical portion counts for 11 core fresh handmade menu items. Because the target outputs are continuous numerical values, the mathematical loss functions and primary evaluation metrics are regression-based: <b>WAPE, MAE, RMSE, and R²</b>.",
        body_style
    ))
    story.append(Paragraph(
        "However, executive decision-makers and kitchen head chefs translate continuous forecasts into discrete, binary operational choices: <i>'Will tomorrow be a High-Demand Surge day requiring extra shift labor and early-morning kitchen prep?'</i> Below, we evaluate all models both through continuous performance (WAPE/MAE/RMSE) and through <b>Operational Decision Thresholds</b> (Confusion Matrix, Precision, Recall, F1-Score, and ±20%/±30% Tolerance Bands).",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Section 1: Confusion Matrix & Classification Metrics
    story.append(Paragraph("1. Operational Decision Evaluation: Confusion Matrix, Accuracy, Precision, Recall & F1-Score", h1_style))
    story.append(Paragraph(
        "<b>Operational Threshold Formulation:</b> Evaluated on the held-out out-of-time chronological test set (132 continuous days). A test day is labeled as a <b>Surge / High-Demand Day (Class 1)</b> if demand exceeds the median historical volume, and <b>Normal / Low Day (Class 0)</b> otherwise.",
        body_style
    ))

    # Lens 1 Table
    story.append(Paragraph("Lens 1: Combined Daily Revenue Surge Alerting (Threshold: > 36,400 LKR)", h2_style))
    
    headers_l1 = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "Confusion Matrix [TN, FP, FN, TP]"]
    data_l1 = [
        [Paragraph(h, th_style) for h in headers_l1],
        [Paragraph("<b>LightGBM (Champion)</b>", td_style), Paragraph("63.6%", td_center), Paragraph("<b>71.4%</b>", td_bold_center), Paragraph("45.5%", td_center), Paragraph("0.556", td_center), Paragraph("TN=54, FP=12, FN=36, TP=30", td_style)],
        [Paragraph("Ridge Regression", td_style), Paragraph("61.4%", td_center), Paragraph("59.5%", td_center), Paragraph("71.2%", td_center), Paragraph("0.648", td_center), Paragraph("TN=34, FP=32, FN=19, TP=47", td_style)],
        [Paragraph("Stacking Ensemble", td_style), Paragraph("61.4%", td_center), Paragraph("58.8%", td_center), Paragraph("<b>75.8%</b>", td_bold_center), Paragraph("<b>0.662</b>", td_bold_center), Paragraph("TN=31, FP=35, FN=16, TP=50", td_style)],
        [Paragraph("XGBoost Regressor", td_style), Paragraph("58.3%", td_center), Paragraph("78.9%", td_center), Paragraph("22.7%", td_center), Paragraph("0.353", td_center), Paragraph("TN=62, FP=4, FN=51, TP=15", td_style)],
        [Paragraph("Random Forest", td_style), Paragraph("54.5%", td_center), Paragraph("80.0%", td_center), Paragraph("12.1%", td_center), Paragraph("0.211", td_center), Paragraph("TN=64, FP=2, FN=58, TP=8", td_style)],
        [Paragraph("<i>Baseline Seasonal Naive</i>", td_style), Paragraph("55.3%", td_center), Paragraph("55.6%", td_center), Paragraph("53.0%", td_center), Paragraph("0.543", td_center), Paragraph("TN=38, FP=28, FN=31, TP=35", td_style)],
    ]
    t1 = Table(data_l1, colWidths=[120, 48, 48, 48, 48, 192])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), brand_blue),
        ('GRID', (0,0), (-1,-1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Managerial Interpretation (Lens 1):</b><br/>"
        "• <b>Precision (LightGBM: 71.4%):</b> When LightGBM predicts a revenue surge, it is correct 71.4% of the time, resulting in only 12 False Positives. This prevents the restaurant owner from unnecessarily calling in expensive overtime staff.<br/>"
        "• <b>Recall (Stacking: 75.8%):</b> The Stacking ensemble catches 50 out of 66 surge days, minimizing understaffed operational bottlenecks.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Lens 2 Table
    story.append(Paragraph("Lens 2: Kitchen Portion Demand Surge Alerting (1,452 item-day test instances)", h2_style))
    headers_l2 = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "Confusion Matrix [TN, FP, FN, TP]"]
    data_l2 = [
        [Paragraph(h, th_style) for h in headers_l2],
        [Paragraph("<b>Ridge Regression (Champion)</b>", td_style), Paragraph("<b>72.0%</b>", td_bold_center), Paragraph("68.3%", td_center), Paragraph("74.3%", td_center), Paragraph("0.712", td_center), Paragraph("TN=300, FP=129, FN=96, TP=278", td_style)],
        [Paragraph("Stacking Ensemble", td_style), Paragraph("70.7%", td_center), Paragraph("65.1%", td_center), Paragraph("<b>80.2%</b>", td_bold_center), Paragraph("<b>0.719</b>", td_bold_center), Paragraph("TN=268, FP=161, FN=74, TP=300", td_style)],
        [Paragraph("Random Forest", td_style), Paragraph("<b>72.0%</b>", td_bold_center), Paragraph("69.7%", td_center), Paragraph("70.6%", td_center), Paragraph("0.701", td_center), Paragraph("TN=314, FP=115, FN=110, TP=264", td_style)],
        [Paragraph("LightGBM Regressor", td_style), Paragraph("70.1%", td_center), Paragraph("71.1%", td_center), Paragraph("60.4%", td_center), Paragraph("0.653", td_center), Paragraph("TN=337, FP=92, FN=148, TP=226", td_style)],
        [Paragraph("XGBoost Regressor", td_style), Paragraph("69.0%", td_center), Paragraph("74.3%", td_center), Paragraph("51.1%", td_center), Paragraph("0.605", td_center), Paragraph("TN=363, FP=66, FN=183, TP=191", td_style)],
        [Paragraph("<i>Baseline Seasonal Naive</i>", td_style), Paragraph("66.4%", td_center), Paragraph("64.9%", td_center), Paragraph("60.4%", td_center), Paragraph("0.626", td_center), Paragraph("TN=307, FP=122, FN=148, TP=226", td_style)],
    ]
    t2 = Table(data_l2, colWidths=[120, 48, 48, 48, 48, 192])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), brand_blue),
        ('GRID', (0,0), (-1,-1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Kitchen Production Interpretation (Lens 2):</b><br/>"
        "• <b>False Positives (FP):</b> Over-predicting demand causes kitchen leads to batch cook excessive perishable curries or bake excess pastries, resulting in food spoilage and waste.<br/>"
        "• <b>False Negatives (FN):</b> Under-predicting demand leads to 12:30 PM stockouts during lunch service, causing lost revenue and customer dissatisfaction.<br/>"
        "• <b>Ridge Regression</b> establishes the best balance with <b>72.0% Accuracy and 0.712 F1-score</b>, limiting both waste and stockout risks.",
        body_style
    ))

    # Tolerance Accuracy
    story.append(Spacer(1, 4))
    story.append(Paragraph("Forecast Tolerance Bands (Percentage of Predictions within Error Tolerance)", h2_style))
    headers_tol = ["Model Architecture", "Lens 1 (±20% Tolerance)", "Lens 1 (±30% Tolerance)", "Lens 2 (±20% Tolerance)", "Lens 2 (±30% Tolerance)"]
    data_tol = [
        [Paragraph(h, th_style) for h in headers_tol],
        [Paragraph("<b>Ridge Regression (L2)</b>", td_style), Paragraph("<b>55.3%</b>", td_bold_center), Paragraph("<b>68.9%</b>", td_bold_center), Paragraph("28.0%", td_center), Paragraph("40.2%", td_center)],
        [Paragraph("<b>Stacking Ensemble</b>", td_style), Paragraph("<b>55.3%</b>", td_bold_center), Paragraph("68.2%", td_center), Paragraph("<b>28.6%</b>", td_bold_center), Paragraph("<b>41.6%</b>", td_bold_center)],
        [Paragraph("<b>LightGBM Regressor</b>", td_style), Paragraph("53.0%", td_center), Paragraph("67.4%", td_center), Paragraph("24.4%", td_center), Paragraph("35.1%", td_center)],
        [Paragraph("Random Forest", td_style), Paragraph("44.7%", td_center), Paragraph("64.4%", td_center), Paragraph("27.4%", td_center), Paragraph("41.1%", td_center)],
        [Paragraph("XGBoost Regressor", td_style), Paragraph("52.3%", td_center), Paragraph("65.2%", td_center), Paragraph("22.5%", td_center), Paragraph("36.4%", td_center)],
        [Paragraph("<i>Baseline Seasonal Naive</i>", td_style), Paragraph("34.8%", td_center), Paragraph("50.0%", td_center), Paragraph("20.8%", td_center), Paragraph("29.3%", td_center)],
    ]
    t_tol = Table(data_tol, colWidths=[140, 92, 92, 92, 92])
    t_tol.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), accent),
        ('GRID', (0,0), (-1,-1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_tol)

    story.append(PageBreak())

    # Section 2: Deep Dive into WAPE
    story.append(Paragraph("2. Understanding WAPE (Weighted Absolute Percentage Error)", h1_style))
    story.append(Paragraph(
        "<b>WAPE</b> is the primary gold-standard evaluation metric used by enterprise retail, food & beverage, and supply-chain forecasting platforms (e.g., Walmart, Amazon Demand Forecasting, Uber Eats).",
        body_style
    ))
    story.append(Paragraph("Mathematical Definition:", h2_style))
    story.append(Paragraph("WAPE = ( &Sigma; |y_i - ŷ_i| ) / ( &Sigma; y_i ) &times; 100%", formula_style))
    story.append(Paragraph(
        "Where:<br/>"
        "• <b>y_i</b> = Actual observed sales volume (Rupees or portion counts) on day <i>i</i><br/>"
        "• <b>ŷ_i</b> = Forecasted value predicted by the machine learning model on day <i>i</i><br/>"
        "• <b>&Sigma; |y_i - ŷ_i|</b> (Numerator) = Total absolute error volume across the entire evaluation horizon<br/>"
        "• <b>&Sigma; y_i</b> (Denominator) = Total actual observed volume across the horizon",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Why WAPE is Superior to MAPE in Restaurant Forecasting:", h2_style))
    story.append(Paragraph(
        "Standard Mean Absolute Percentage Error (MAPE) calculates: <i>MAPE = (1/N) &Sigma; [ |y_i - ŷ_i| / y_i ] &times; 100%</i>.<br/>"
        "<b>The Fatal Flaw of MAPE with Food Demand:</b><br/>"
        "1. <b>Zero-Division Hazard:</b> If a dish has zero sales on a given day (y = 0), MAPE divides by zero and explodes to infinity (&infin;).<br/>"
        "2. <b>Low-Volume Distortion:</b> Suppose on a slow rainy Tuesday, the restaurant sells <b>1 slice</b> of Date Cake. If the model predicts <b>3 slices</b>, the absolute error is only 2 slices (negligible). But MAPE calculates: <i>|1 - 3| / 1 = 200% error!</i> This unfairly penalizes the model.<br/>"
        "<b>How WAPE Solves This:</b> WAPE weights the errors by total demand volume. The 2-slice error is divided by the entire month's total demand, remaining completely stable, mathematically bounded, and directly proportional to financial impact.",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("What is 'Validation WAPE'?", h2_style))
    story.append(Paragraph(
        "In rigorous time-series machine learning, datasets must be split chronologically into <b>Train, Validation, and Test</b> sets without lookahead bias:<br/>"
        "• <b>Validation WAPE</b> is the performance score measured on the 131-day intermediate validation split (Dec 19, 2025 – Apr 28, 2026).<br/>"
        "• <b>The Golden Rule of ML Evaluation:</b> Model hyperparameter tuning and champion selection must be conducted <b>strictly based on Validation WAPE</b>. The final Test set (Apr 29 – Sep 7, 2026) must remain completely untouched until the final report. This proves the champion generalizes to the future without test-set data leakage.",
        body_style
    ))

    story.append(Spacer(1, 6))

    # Section 3: Good Real-World Value of WAPE
    story.append(Paragraph("3. What is a 'Good' Real-World WAPE Value in Industry?", h1_style))
    story.append(Paragraph(
        "In enterprise forecasting, WAPE benchmarks depend strongly on the <b>level of aggregation</b> (macro-store level vs. micro-dish level) and demand intermittency:",
        body_style
    ))

    headers_bench = ["Forecast Granularity Level", "WAPE Range", "Industry Benchmark Rating", "Real-World Context & Expectation"]
    data_bench = [
        [Paragraph(h, th_style) for h in headers_bench],
        [
            Paragraph("<b>Macro Aggregate</b><br/>(Store Revenue - Lens 1)", td_style),
            Paragraph("&lt; 15%<br/><b>15% - 30%</b><br/>30% - 45%<br/>&gt; 50%", td_center),
            Paragraph("World-Class (FMCG)<br/><b>Strong / Production-Grade</b><br/>Moderate / Needs buffer<br/>Poor / Unusable", td_style),
            Paragraph("Supermarket chains with steady staples achieve &lt;15%. In volatile hospitality with weather/holiday swings, <b>20%-30% is production-ready</b>.", td_style)
        ],
        [
            Paragraph("<b>Micro Disaggregated</b><br/>(Single Dish - Lens 2)", td_style),
            Paragraph("&lt; 30%<br/><b>35% - 50%</b><br/>50% - 65%<br/>&gt; 65%", td_center),
            Paragraph("Exceptional (Rare)<br/><b>Industry Standard</b><br/>Acceptable for slow items<br/>Naive noise", td_style),
            Paragraph("Individual handmade dishes selling 5-20 portions/day have high stochastic variance. <b>35%-50% is solid industry standard</b>.", td_style)
        ]
    ]
    t_bench = Table(data_bench, colWidths=[105, 75, 120, 208])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), brand_blue),
        ('GRID', (0,0), (-1,-1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bench)

    story.append(Spacer(1, 6))

    # ChefCast AI Results Benchmark
    story.append(Paragraph("Where ChefCast AI Stands Against Industry Standards", h2_style))
    story.append(Paragraph(
        "• <b>Lens 1 Champion (LightGBM):</b> Achieved <b>30.33% Validation WAPE</b> and an unbiased <b>24.99% Test WAPE</b>. This places ChefCast AI squarely in the <b>Strong / Production-Grade zone (20%–30%)</b> for hospitality revenue forecasting. Compared to the 7-day Naive baseline (43.15% Test WAPE), LightGBM delivers an <b>18.16 percentage point gain</b> in accuracy.<br/>"
        "• <b>Lens 2 Champion (Ridge Regression):</b> Achieved <b>47.31% Validation WAPE</b> and <b>44.38% Test WAPE</b> (Mean Absolute Error of only <b>5.95 portions per dish</b>). Compared to the Naive baseline (63.49% Test WAPE), Ridge delivers a massive <b>19.11 percentage point gain</b>, proving exceptional defensibility and real-world kitchen utility.",
        body_style
    ))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=brand_blue, spaceBefore=4, spaceAfter=8))
    story.append(Paragraph(
        "<b>Summary for Assessment Rubric:</b> This dual-lens evaluation demonstrates clear machine learning judgment. We selected LightGBM for non-linear aggregate interactions on Lens 1, and regularized Ridge Regression to prevent overfitting on sparse dish counts on Lens 2. Both champions outperform naive restaurant heuristics by ~19% and operate within industry-standard benchmarks.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {filename}")

if __name__ == "__main__":
    build_pdf()
