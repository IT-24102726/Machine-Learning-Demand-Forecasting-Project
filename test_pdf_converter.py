import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Preformatted
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
            header_text = getattr(self, "doc_header_text", "ChefCast AI — SLIIT IT3091")
            self.drawString(45, 805, header_text)
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(45, 798, 550, 798)
        
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(550, 30, text)
        footer_sub = getattr(self, "doc_footer_sub", "SLIIT IT3091 Machine Learning | Group 2026-AI-36")
        self.drawString(45, 30, footer_sub)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(45, 42, 550, 42)
        self.restoreState()

def clean_inline_md(text):
    """Safely converts markdown inline formatting into ReportLab HTML tags."""
    if not text:
        return ""
    
    # 1. Escape XML characters first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 2. Math expressions: $...$ -> <i><font color="#1E3A8A">...</font></i>
    def math_repl(m):
        raw = m.group(1).replace(r"\sum", "Σ").replace(r"\times", "×").replace(r"\cdot", "·")
        raw = raw.replace(r"\quad", "  ").replace(r"\text", "")
        raw = raw.replace(r"\{", "{").replace(r"\}", "}")
        raw = raw.replace(r"\_", "_")
        return f'<b><i><font color="#1E3A8A">{raw}</font></i></b>'
    text = re.sub(r'\$(.*?)\$', math_repl, text)
    
    # 3. Bold-italic: ***text***
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    
    # 4. Bold: **text**
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # 5. Italic: *text* (avoiding lone asterisks)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    
    # 6. Inline code: `code`
    text = re.sub(r'`(.*?)`', r'<font face="Courier" color="#0369A1"><b>\1</b></font>', text)
    
    return text

def parse_markdown_to_flowables(md_content, base_title=""):
    styles = getSampleStyleSheet()
    
    brand_navy = colors.HexColor("#0F172A")
    brand_blue = colors.HexColor("#1E3A8A")
    accent_blue = colors.HexColor("#0284C7")
    slate_text = colors.HexColor("#1E293B")
    
    h1_style = ParagraphStyle(
        'MD_H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=brand_navy,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'MD_H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=brand_blue,
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h3_style = ParagraphStyle(
        'MD_H3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14.5,
        textColor=accent_blue,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    h4_style = ParagraphStyle(
        'MD_H4',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13.5,
        textColor=brand_navy,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'MD_Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=slate_text,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'MD_Bullet',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'MD_Callout',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )
    
    th_style = ParagraphStyle(
        'MD_TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=10,
        textColor=colors.white,
        alignment=0
    )
    
    td_style = ParagraphStyle(
        'MD_TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=slate_text,
        alignment=0
    )
    
    code_block_style = ParagraphStyle(
        'MD_CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#0F172A")
    )

    flowables = []
    lines = md_content.split('\n')
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i]
        stripped = line.strip()
        
        # Empty line
        if not stripped:
            i += 1
            continue
            
        # Code block
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1 # skip closing ```
            code_text = "\n".join(code_lines)
            code_safe = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_p = Paragraph(f"<pre>{code_safe}</pre>", code_block_style)
            t = Table([[code_p]], colWidths=[505])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            flowables.append(Spacer(1, 3))
            flowables.append(t)
            flowables.append(Spacer(1, 4))
            continue
            
        # Horizontal rule
        if stripped in ["---", "***", "___"]:
            flowables.append(Spacer(1, 4))
            flowables.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#CBD5E1"), spaceAfter=6, spaceBefore=4))
            i += 1
            continue
            
        # Table
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            
            # Parse table rows
            raw_rows = []
            for tl in table_lines:
                # check if it's separator row: |---|---|
                inner = tl.strip('|')
                parts = [p.strip() for p in inner.split('|')]
                if all(re.match(r'^:?-+:?$', p) for p in parts if p):
                    continue
                raw_rows.append(parts)
            
            if raw_rows:
                num_cols = max(len(r) for r in raw_rows)
                # Normalize length
                for r in raw_rows:
                    while len(r) < num_cols:
                        r.append("")
                
                # Build flowable rows
                table_flowables = []
                for row_idx, r in enumerate(raw_rows):
                    row_cells = []
                    for c in r:
                        c_clean = clean_inline_md(c)
                        if row_idx == 0:
                            row_cells.append(Paragraph(c_clean, th_style))
                        else:
                            row_cells.append(Paragraph(c_clean, td_style))
                    table_flowables.append(row_cells)
                
                # Column widths allocation
                avail_width = 505.0
                col_width = avail_width / num_cols
                
                t = Table(table_flowables, colWidths=[col_width]*num_cols)
                t_style = [
                    ('BACKGROUND', (0,0), (-1,0), brand_navy),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ('TOPPADDING', (0,0), (-1,-1), 4),
                    ('LEFTPADDING', (0,0), (-1,-1), 4),
                    ('RIGHTPADDING', (0,0), (-1,-1), 4),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
                ]
                for r_i in range(1, len(table_flowables)):
                    if r_i % 2 == 1:
                        t_style.append(('BACKGROUND', (0, r_i), (-1, r_i), colors.HexColor("#F8FAFC")))
                    else:
                        t_style.append(('BACKGROUND', (0, r_i), (-1, r_i), colors.white))
                t.setStyle(TableStyle(t_style))
                flowables.append(Spacer(1, 4))
                flowables.append(t)
                flowables.append(Spacer(1, 5))
            continue
            
        # Headings
        if stripped.startswith("#### "):
            h_text = clean_inline_md(stripped[5:].strip())
            flowables.append(Paragraph(h_text, h4_style))
            i += 1
            continue
        elif stripped.startswith("### "):
            h_text = clean_inline_md(stripped[4:].strip())
            flowables.append(Paragraph(h_text, h3_style))
            i += 1
            continue
        elif stripped.startswith("## "):
            h_text = clean_inline_md(stripped[3:].strip())
            flowables.append(Paragraph(h_text, h2_style))
            i += 1
            continue
        elif stripped.startswith("# "):
            h_text = clean_inline_md(stripped[2:].strip())
            flowables.append(Paragraph(h_text, h1_style))
            i += 1
            continue
            
        # Blockquote
        if stripped.startswith("> "):
            bq_text = clean_inline_md(stripped[2:].strip())
            bq_p = Paragraph(bq_text, callout_style)
            t = Table([[bq_p]], colWidths=[505])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
                ('LINEBEFORE', (0,0), (0,-1), 3, brand_blue),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            flowables.append(Spacer(1, 2))
            flowables.append(t)
            flowables.append(Spacer(1, 3))
            i += 1
            continue
            
        # Unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            item_text = clean_inline_md(stripped[2:].strip())
            p_text = f"&bull;&nbsp;&nbsp;{item_text}"
            flowables.append(Paragraph(p_text, bullet_style))
            i += 1
            continue
            
        # Ordered list
        m_num = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if m_num:
            num = m_num.group(1)
            item_text = clean_inline_md(m_num.group(2))
            p_text = f"<b>{num}.</b>&nbsp;&nbsp;{item_text}"
            flowables.append(Paragraph(p_text, bullet_style))
            i += 1
            continue
            
        # Standard paragraph
        p_text = clean_inline_md(stripped)
        flowables.append(Paragraph(p_text, body_style))
        i += 1
        
    return flowables

def build_pdf_document(input_md_path, output_pdf_path, header_title, footer_sub="SLIIT IT3091 Machine Learning | Group 2026-AI-36"):
    with open(input_md_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=48,
        bottomMargin=45
    )
    
    # Custom canvas callback wrapper
    def canvas_maker(*args, **kwargs):
        c = NumberedCanvas(*args, **kwargs)
        c.doc_header_text = header_title
        c.doc_footer_sub = footer_sub
        return c
        
    flowables = parse_markdown_to_flowables(content, header_title)
    doc.build(flowables, canvasmaker=canvas_maker)
    print(f"Successfully generated: {output_pdf_path} (size: {os.path.getsize(output_pdf_path)} bytes)")

if __name__ == "__main__":
    test_md = "docs/individual_PLJ_IT24102572.md"
    test_pdf = "docs/ChefCast_AI_Individual_PLJ_IT24102572.pdf"
    build_pdf_document(
        test_md,
        test_pdf,
        "ChefCast AI — Personal Learning Journey | IT24102572"
    )
