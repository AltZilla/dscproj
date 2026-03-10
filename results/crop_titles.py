import os
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = SCRIPT_DIR / "graphs"

def get_crop_amounts():
    # Adjust these values to crop less
    png_crop_pixels = 85   # Was 150
    pdf_crop_points = 20   # Was 35
    return png_crop_pixels, pdf_crop_points

def crop_png(file_path):
    png_crop, _ = get_crop_amounts()
    img = Image.open(file_path)
    width, height = img.size
    # crop box is (left, upper, right, lower)
    cropped_img = img.crop((0, png_crop, width, height))
    cropped_img.save(file_path)
    print(f"Cropped PNG: {file_path.name}")

def crop_pdf(file_path):
    _, pdf_crop = get_crop_amounts()
    reader = PdfReader(file_path)
    writer = PdfWriter()
    for page in reader.pages:
        page.mediabox.top = float(page.mediabox.top) - pdf_crop
        writer.add_page(page)
    
    with open(file_path, "wb") as f:
        writer.write(f)
    print(f"Cropped PDF: {file_path.name}")

if __name__ == "__main__":
    count = 0
    for file in GRAPHS_DIR.iterdir():
        if not file.name.startswith("fig"):
            continue
            
        if file.suffix == ".png":
            crop_png(file)
            count += 1
        elif file.suffix == ".pdf":
            crop_pdf(file)
            count += 1
            
    print(f"Successfully cropped {count} files.")
