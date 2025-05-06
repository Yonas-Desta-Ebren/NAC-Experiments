import os
import sys
import subprocess
import tempfile

def install_required_packages():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "weasyprint"])
        return True
    except subprocess.CalledProcessError:
        return False

def markdown_to_html(markdown_file):
    try:
        import markdown
        with open(markdown_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Model Comparison Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                img {{
                    max-width: 100%;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            {markdown.markdown(md_content, extensions=['tables'])}
        </body>
        </html>
        """
        
        return html_content
    except Exception as e:
        print(f"Error converting Markdown to HTML: {e}")
        return None

def html_to_pdf(html_content, output_file):
    try:
        from weasyprint import HTML
        
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
            f.write(html_content)
            temp_html_file = f.name
        
        # Convert HTML to PDF
        HTML(filename=temp_html_file).write_pdf(output_file)
        
        # Remove the temporary file
        os.unlink(temp_html_file)
        
        return True
    except Exception as e:
        print(f"Error converting HTML to PDF: {e}")
        return False

def main():
    markdown_file = 'model_comparison_report.md'
    output_file = 'model_comparison_report.pdf'
    
    if not os.path.exists(markdown_file):
        print(f"Error: {markdown_file} not found")
        return
    
    print("Installing required packages...")
    if not install_required_packages():
        print("Failed to install required packages. Please install them manually:")
        print("pip install markdown weasyprint")
        return
    
    print("Converting Markdown to HTML...")
    html_content = markdown_to_html(markdown_file)
    if not html_content:
        return
    
    print("Converting HTML to PDF...")
    if html_to_pdf(html_content, output_file):
        print(f"Successfully created {output_file}")
    else:
        print("Failed to create PDF")

if __name__ == "__main__":
    main()
