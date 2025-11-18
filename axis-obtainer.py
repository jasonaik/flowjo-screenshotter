from PIL import Image, ImageDraw

# Load your square image
img = Image.open("templates/image.png").convert("RGBA")
w, h = img.size

# Create a mask
mask = Image.new("L", (w, h), 0)  # fully transparent to start
draw = ImageDraw.Draw(mask)

# Thickness of the L arms (adjust as needed)
thickness = int(0.1 * w)  # 25% of size

# Horizontal bar along bottom
draw.rectangle([0, h - thickness, w, h], fill=255)

# Vertical bar along left
draw.rectangle([0, 0, thickness+3, h], fill=255)

# Apply mask as alpha channel
result = img.copy()
result.putalpha(mask)

result.save("templates/axis-template.png")
