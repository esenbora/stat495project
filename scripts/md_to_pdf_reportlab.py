from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import re
import os

def convert_md_to_pdf(source_md, output_pdf):
    doc = SimpleDocTemplate(output_pdf, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Caption', parent=styles['Italic'], alignment=1)) # Center alignment
    
    story = []
    
    with open(source_md, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Simple state machine or line-by-line processing
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 12))
            continue
        
        # Header 1
        if line.startswith('# '):
            story.append(Paragraph(line[2:], styles['Heading1']))
            story.append(Spacer(1, 12))
        
        # Header 2
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['Heading2']))
            story.append(Spacer(1, 10))
            
        # Header 3
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading3']))
            story.append(Spacer(1, 8))
            
        # List item
        elif line.startswith('- '):
            # Use bullet character
            text = f'&bull; {line[2:]}'
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 4))
            
        # Image
        elif line.startswith('![') and '](' in line:
            match = re.search(r'\!\[(.*?)\]\((.*?)\)', line)
            if match:
                caption = match.group(1)
                img_path = match.group(2)
                
                # Resolve relative path
                base_dir = os.path.dirname(source_md)
                full_img_path = os.path.join(base_dir, img_path)
                
                if os.path.exists(full_img_path):
                    try:
                        # Create ReportLab Image
                        # Restrict width to page width - margins (approx 6 inches)
                        img = RLImage(full_img_path)
                        
                        # Resize if too wide
                        max_width = 6 * inch
                        img_width = img.drawWidth
                        img_height = img.drawHeight
                        
                        if img_width > max_width:
                            ratio = max_width / img_width
                            img.drawWidth = max_width
                            img.drawHeight = img_height * ratio
                        
                        story.append(img)
                        story.append(Spacer(1, 6))
                        story.append(Paragraph(caption, styles['Caption']))
                        story.append(Spacer(1, 12))
                        
                    except Exception as e:
                        print(f"Error adding image {full_img_path}: {e}")
                else:
                    print(f"Image not found: {full_img_path}")
        
        # Normal text
        else:
            story.append(Paragraph(line, styles['BodyText']))
            story.append(Spacer(1, 6))

    doc.build(story)
    print(f"Successfully created {output_pdf}")

if __name__ == "__main__":
    source = '/Users/boraesen/Desktop/stat495project/reports/eda_walkthrough.md'
    output = '/Users/boraesen/Desktop/stat495project/reports/eda_walkthrough.pdf'
    convert_md_to_pdf(source, output)
