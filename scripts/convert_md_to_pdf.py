import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    # Read Markdown file
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert to HTML
    html_content = markdown.markdown(text)

    # Add some basic styling
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, sans-serif; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert HTML to PDF
    with open(output_pdf, "wb") as result_file:
        # pisa.CreatePDF expects the base path to resolve relative links (images)
        pisa_status = pisa.CreatePDF(
            html_content,
            dest=result_file,
            path=source_md  # Use source file path as base for relative links
        )

    if pisa_status.err:
        print(f"Error converting {source_md} to PDF")
        return False
    else:
        print(f"Successfully created {output_pdf}")
        return True

if __name__ == "__main__":
    source = '/Users/boraesen/Desktop/stat495project/reports/eda_walkthrough.md'
    output = '/Users/boraesen/Desktop/stat495project/reports/eda_walkthrough.pdf'
    convert_md_to_pdf(source, output)
