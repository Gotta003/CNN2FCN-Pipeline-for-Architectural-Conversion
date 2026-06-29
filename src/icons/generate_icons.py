import os
from PIL import Image, ImageDraw, ImageFont

PHOTO_PATH="assets/icons/"
FONT_NAME = "DejaVuSans.ttf"

def unicode_to_png(unicode_char, output_filename, font_path=FONT_NAME, font_size=120, color="#555555"):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    image_size=int(font_size*1.2)
    img=Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
    draw=ImageDraw.Draw(img)
    try:
        font=ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Could not find font at `{font_path}`")
        return
    
    bbox=draw.textbbox((0,0), unicode_char, font=font)
    text_width=bbox[2]-bbox[0]
    text_height=bbox[3]-bbox[1]
    x=(image_size-text_width)/2
    y=(image_size-text_height)/2
    draw.text((x, y-bbox[1]), unicode_char, font=font, fill=color)
    img.save(output_filename)
    print(f"Created: {output_filename}")
    
if __name__=="__main__":
    WHITE="#CCCCCC"
    unicode_to_png("⬡", f"{PHOTO_PATH}hexagon_light.png", color="#333333")
    unicode_to_png("⬡", f"{PHOTO_PATH}hexagon_dark.png", color=WHITE)
    unicode_to_png("⬡", f"{PHOTO_PATH}hexagon_warning.png", color="#EF9F27")
    unicode_to_png("✓", f"{PHOTO_PATH}check.png", color="#1D9E75")
    unicode_to_png("■", f"{PHOTO_PATH}stop.png", color="#333333")
    unicode_to_png("⟳", f"{PHOTO_PATH}refresh.png", color=WHITE)
    unicode_to_png("▤", f"{PHOTO_PATH}dataset.png", color=WHITE)
    unicode_to_png("⚙", f"{PHOTO_PATH}config.png", color=WHITE)
    unicode_to_png("☰", f"{PHOTO_PATH}pipeline.png", color=WHITE)
    unicode_to_png("▶", f"{PHOTO_PATH}run.png", color=WHITE)
    unicode_to_png("⚡", f"{PHOTO_PATH}results.png", color=WHITE)
    unicode_to_png("↗", f"{PHOTO_PATH}redirect.png", color=WHITE)
    unicode_to_png("○", f"{PHOTO_PATH}circle.png", color=WHITE)
    unicode_to_png("✗", f"{PHOTO_PATH}error.png", color="#892616")
    unicode_to_png("–", f"{PHOTO_PATH}skip.png", color=WHITE)