from fpdf import FPDF
from PIL import Image
import re
import os
import tempfile

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'AFAD Historical Data EDA', border=False, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def convert_md_to_pdf(source_md, output_pdf):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    with open(source_md, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
        
        # Header 1
        if line.startswith('# '):
            pdf.set_font('helvetica', 'B', 24)
            pdf.cell(0, 10, line[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
        
        # Header 2
        elif line.startswith('## '):
            pdf.set_font('helvetica', 'B', 18)
            pdf.cell(0, 10, line[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
            
        # Header 3
        elif line.startswith('### '):
            pdf.set_font('helvetica', 'B', 14)
            pdf.cell(0, 10, line[4:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            
        # List item
        elif line.startswith('- '):
            pdf.set_font('helvetica', '', 12)
            pdf.cell(10) # Indent
            pdf.cell(0, 8, f'- {line[2:]}', new_x="LMARGIN", new_y="NEXT")
            
        # Image
        elif line.startswith('![') and '](' in line:
            print(f"Found image line: {line}")
            match = re.search(r'\!\[(.*?)\]\((.*?)\)', line)
            if match:
                caption = match.group(1)
                img_path = match.group(2)
                print(f"Extracted image path: {img_path}")
                
                # Resolve relative path
                base_dir = os.path.dirname(source_md)
                full_img_path = os.path.join(base_dir, img_path)
                print(f"Full image path: {full_img_path}")
                
                if os.path.exists(full_img_path):
                    try:
                        # Convert to RGB to avoid transparency issues
                        with Image.open(full_img_path) as img:
                            rgb_img = img.convert('RGB')
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
                                rgb_img.save(tmp_img.name, 'JPEG', quality=90)
                                tmp_img_name = tmp_img.name
                            
                            print(f"Adding image to PDF: {tmp_img_name}")
                            pdf.image(tmp_img_name, w=170)
                            os.unlink(tmp_img_name) # Clean up temp file

                        pdf.ln(2)
                        pdf.set_font('helvetica', 'I', 10)
                        pdf.cell(0, 8, caption, align='C', new_x="LMARGIN", new_y="NEXT")
                        pdf.ln(5)
                    except Exception as e:
                        print(f"Error adding image {full_img_path}: {e}")
                else:
                    print(f"Image not found: {full_img_path}")
        
        # Normal text
        else:
            pdf.set_font('helvetica', '', 12)
            pdf.multi_cell(0, 8, line)

    pdf.output(output_pdf)
    print(f"Successfully created {output_pdf}")

if __name__ == "__main__":
    source = '/Users/boraesen/Desktop/stat495project/reports/eda_walkthrough.md'
    output = '/Users/boraesen/Desktop/stat495project/reports/eda_walkthrough.pdf'
    convert_md_to_pdf(source, output)
