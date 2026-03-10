import os
from pathlib import Path
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = SCRIPT_DIR / "graphs"

# Configured manually for each figure to ensure no clipping of the actual graph box!
# Values: (PNG pixels to crop from top, PDF points to crop from top)
# A 300 DPI PNG with 140 pixels ~ 33.6 points in 72 DPI PDF.
CROP_CONFIG = {
    "fig01_vae_recon_loss": (140, 33),
    "fig02_kl_divergence": (140, 33),
    "fig03_fedgru_mae": (140, 33),
    "fig04_fl_rmse": (140, 33),
    "fig05_privacy_accuracy": (140, 33),
    "fig06_stackelberg_price": (140, 33),
    "fig07_hourly_price_demand": (140, 33),
    "fig08_de_fitness": (140, 33),
    "fig09_cost_reduction": (140, 33),
    "fig10_par_comparison": (140, 33),
    "fig11_load_distribution": (140, 33),
    "fig12_radar_comparison": (160, 38) # Radar chart has extra padding (pad=20) internally
}

def crop_png(file_path, crop_pixels):
    img = Image.open(file_path)
    width, height = img.size
    # crop box is (left, upper, right, lower)
    # This leaves a small margin above the top axis spine
    cropped_img = img.crop((0, crop_pixels, width, height))
    cropped_img.save(file_path)

def crop_pdf(file_path, crop_points):
    reader = PdfReader(file_path)
    writer = PdfWriter()
    for page in reader.pages:
        # Reduce the top of the mediabox coordinates
        page.mediabox.top = float(page.mediabox.top) - crop_points
        writer.add_page(page)
    
    with open(file_path, "wb") as f:
        writer.write(f)

if __name__ == "__main__":
    png_count, pdf_count = 0, 0
    
    for file in GRAPHS_DIR.iterdir():
        if not file.name.startswith("fig"):
            continue
            
        base_name = file.stem
        if base_name not in CROP_CONFIG:
            continue
            
        crop_px, crop_pt = CROP_CONFIG[base_name]
        
        if file.suffix == ".png":
            crop_png(file, crop_px)
            png_count += 1
        elif file.suffix == ".pdf":
            crop_pdf(file, crop_pt)
            pdf_count += 1
            
    print(f"Successfully cropped {png_count} PNGs and {pdf_count} PDFs.")
    print("The graphs have been updated in place.")
