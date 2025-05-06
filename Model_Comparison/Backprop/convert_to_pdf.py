import os
import sys
import subprocess
import platform

def check_npm_installed():
    try:
        subprocess.run(['npm', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_markdown_pdf_installed():
    try:
        subprocess.run(['markdown-pdf', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_markdown_pdf():
    try:
        subprocess.run(['npm', 'install', '-g', 'markdown-pdf'], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def convert_markdown_to_pdf(markdown_file, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(markdown_file)[0] + '.pdf'
    
    try:
        subprocess.run(['markdown-pdf', markdown_file, '-o', output_file], check=True)
        print(f"Successfully converted {markdown_file} to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}")
        return False

def main():
    markdown_file = 'model_comparison_report.md'
    
    if not os.path.exists(markdown_file):
        print(f"Error: {markdown_file} not found")
        return
    
    if not check_npm_installed():
        print("Error: npm is not installed. Please install Node.js and npm first.")
        return
    
    if not check_markdown_pdf_installed():
        print("markdown-pdf is not installed. Attempting to install...")
        if not install_markdown_pdf():
            print("Failed to install markdown-pdf. Please install it manually with 'npm install -g markdown-pdf'")
            return
    
    convert_markdown_to_pdf(markdown_file)

if __name__ == "__main__":
    main()
