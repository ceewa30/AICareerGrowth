import os
from pathlib import Path
import markdown
from xhtml2pdf import pisa

def export_markdown_to_pdf(markdown_text: str, output_filename: str = "Career_Pathway_Report.pdf") -> str:
    """
    Converts raw markdown text to standardized styled HTML,
    then compiles it natively into a local downloadable PDF file asset.
    """
    try:
        # 1. Convert raw Markdown string into standard HTML structures
        html_body = markdown.markdown(markdown_text)

        # 2. FIXED: Completely removed the problematic print paging subrules (@bottom-right)
        # This completely side-steps the internal xhtml2pdf NotImplementedType parser crash bug.
        complete_html_document = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: letter;
                    margin: 1in;
                }}
                body {{
                    font-family: Helvetica, Arial, sans-serif;
                    color: #2D3748;
                    line-height: 1.6;
                    font-size: 11pt;
                }}
                h1 {{
                    color: #1A365D;
                    font-size: 22pt;
                    border-bottom: 1px solid #E2E8F0;
                    padding-bottom: 6px;
                    margin-bottom: 18px;
                }}
                h2 {{
                    color: #2B6CB0;
                    font-size: 15pt;
                    margin-top: 25px;
                    border-bottom: 1px solid #E2E8F0;
                    padding-bottom: 3px;
                }}
                h3 {{
                    color: #2D3748;
                    font-size: 12pt;
                    margin-top: 18px;
                    background-color: #F7FAFC;
                    padding: 6px;
                    border-left: 3px solid #4299E1;
                }}
                ul {{
                    padding-left: 20px;
                }}
                li {{
                    margin-bottom: 5px;
                }}
                code {{
                    font-family: monospace;
                    background-color: #EDF2F7;
                    padding: 1px 3px;
                    font-size: 10pt;
                }}
                hr {{
                    border: 0;
                    border-top: 1px solid #E2E8F0;
                    margin: 18px 0;
                }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """

        # 3. Resolve local export destination folder paths safely
        export_dir = Path(__file__).resolve().parent.parent / "exports"
        os.makedirs(export_dir, exist_ok=True)
        pdf_file_path = export_dir / output_filename

        # 4. Stream compile transaction
        with open(pdf_file_path, "wb") as output_file:
            pisa_status = pisa.CreatePDF(complete_html_document, dest=output_file)

        if pisa_status.err:
            raise Exception("Internal PDF compiler structure sub-layer failure.")

        return str(pdf_file_path)

    except Exception as e:
        print(f"❌ PDF Export Module Failed: {e}")
        return ""
