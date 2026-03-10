import os
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = SCRIPT_DIR / "graphs"

def find_top_spine(img_path):
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    pixels = img.load()
    
    start_x = int(width * 0.2)
    end_x = int(width * 0.8)
    check_width = end_x - start_x
    
    # scan from top down
    for y in range(50, int(height * 0.3)):
        dark_pixels = 0
        for x in range(start_x, end_x):
            r, g, b = pixels[x, y]
            # axes spines are usually black/very dark gray
            if r < 50 and g < 50 and b < 50:
                dark_pixels += 1
                
        # if >80% are dark, we found the horizontal spine!
        if dark_pixels > check_width * 0.8:
            return y
            
    return None

if __name__ == "__main__":
    for file in GRAPHS_DIR.iterdir():
        if file.suffix == ".png" and file.name.startswith("fig"):
            y = find_top_spine(file)
            print(f"{file.name}: {y}")
