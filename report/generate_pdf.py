import markdown
import os
import re
import subprocess
from PIL import Image
import shutil

# Create temp directory for resized images
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
temp_img_dir = os.path.join(base_dir, 'temp_images')
os.makedirs(temp_img_dir, exist_ok=True)

def resize_image(src_path, max_width=1200):
    """Resize image if wider than max_width, save to temp dir."""
    rel_path = os.path.relpath(src_path, root_dir)
    temp_path = os.path.join(temp_img_dir, rel_path.replace(os.sep, '_'))
    
    try:
        with Image.open(src_path) as img:
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.LANCZOS)
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(temp_path, 'JPEG', quality=90)
        return temp_path
    except Exception as e:
        print(f"Warning: could not resize {src_path}: {e}")
        return src_path

# Read markdown
with open(os.path.join(base_dir, 'report.md'), 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML
html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

# Fix image paths and resize images
def fix_image_path(match):
    alt = match.group(1)
    path = match.group(2)
    abs_path = os.path.normpath(os.path.join(base_dir, path))
    if os.path.exists(abs_path):
        resized = resize_image(abs_path)
        return '<img alt="{}" src="file:///{}" style="max-width:100%;height:auto;display:block;margin:12pt auto;" />'.format(
            alt, resized.replace(chr(92), "/"))
    else:
        return match.group(0)

html_body = re.sub(r'<img alt="([^"]+)" src="([^"]+)" />', fix_image_path, html_body)

# Wrap in full HTML with CSS
html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ML Assignment 2 Report</title>
<style>
  @page { size: A4; margin: 15mm; }
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    max-width: 180mm;
    margin: 0 auto;
    padding: 5mm 5mm;
  }
  h1 {
    font-size: 20pt;
    color: #1a1a1a;
    border-bottom: 2px solid #333;
    padding-bottom: 8px;
    margin-top: 24pt;
  }
  h2 {
    font-size: 16pt;
    color: #222;
    margin-top: 20pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
  }
  h3 {
    font-size: 13pt;
    color: #333;
    margin-top: 16pt;
  }
  h4 {
    font-size: 12pt;
    color: #444;
    margin-top: 14pt;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 10pt;
    margin: 12pt 0;
    table-layout: fixed;
  }
  th, td {
    border: 1px solid #999;
    padding: 6px 10px;
    text-align: left;
    vertical-align: top;
    word-wrap: break-word;
  }
  th {
    background-color: #f0f0f0;
    font-weight: bold;
  }
  tr:nth-child(even) {
    background-color: #fafafa;
  }
  code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: Consolas, monospace;
    font-size: 10pt;
  }
  pre {
    background-color: #f4f4f4;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    font-family: Consolas, monospace;
    font-size: 9.5pt;
  }
  img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 12pt auto;
    border: 1px solid #ddd;
    border-radius: 4px;
  }
  blockquote {
    border-left: 4px solid #ccc;
    margin: 12pt 0;
    padding: 8px 16px;
    background-color: #f9f9f9;
    color: #555;
  }
  hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 20pt 0;
  }
  ul, ol {
    margin: 8pt 0;
    padding-left: 24pt;
  }
  li {
    margin: 4pt 0;
  }
  p {
    margin: 8pt 0;
  }
  strong {
    color: #111;
  }
  a {
    color: #0066cc;
    text-decoration: none;
  }
</style>
</head>
<body>
''' + html_body + '''
</body>
</html>'''

html_path = os.path.join(base_dir, 'report_temp.html')
pdf_path = os.path.join(base_dir, 'report.pdf')

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)

print('HTML generated:', html_path)

# Find Chrome or Edge
chrome_paths = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

chrome_exe = None
for path in chrome_paths:
    if os.path.exists(path):
        chrome_exe = path
        break

if chrome_exe:
    cmd = [
        chrome_exe,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=15000",
        "--no-pdf-header-footer",
        "--print-to-pdf=" + pdf_path,
        "file:///" + html_path.replace(chr(92), "/"),
    ]
    print("Using:", chrome_exe)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode == 0 and os.path.exists(pdf_path):
        size_mb = os.path.getsize(pdf_path) / (1024*1024)
        print("PDF generated:", pdf_path)
        print("Size: {:.2f} MB".format(size_mb))
    else:
        print("Chrome print failed:", result.stderr)
else:
    print("No Chrome/Edge found for PDF generation")

# Cleanup temp files
if os.path.exists(html_path):
    os.remove(html_path)
if os.path.exists(temp_img_dir):
    shutil.rmtree(temp_img_dir)
