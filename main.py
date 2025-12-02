import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import datetime
import os

# === CONFIG ===
pdf_path = "data/28-11-25/30-Nov-2025-405.pdf"

DPI = 300
MATCH_THRESHOLD = 0.3

# Change these to adjust how much area around a found match is suppressed
# to avoid detecting the same plot multiple times
SUPPRESS_SCALE_X = 1.5
SUPPRESS_SCALE_Y = 1.2

# Change the search padding if your plots are larger and are being truncated
SEARCH_PAD_X=80
SEARCH_PAD_Y=150
# SEARCH_PAD_Y=80

# min area for contour to be considered a plot box; change if frame detection fails
MIN_CONTOUR_AREA = 5000  

FOLDER_NAME = pdf_path.split("\\")[-1]

output_dir = Path(f"plots/png/{datetime.date.today()}/{FOLDER_NAME}")
os.makedirs(output_dir, exist_ok=True)

template_path = "templates/axis-template.png"


# === HELPERS ===
def render_page_to_image(page, dpi=DPI):
    """Render a PyMuPDF page to a BGR OpenCV image."""
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def find_tight_plot_crop(
    img_gray,
    img_color,
    x0,
    y0,
    tw,
    th,
    search_pad_x=SEARCH_PAD_X,
    search_pad_y=SEARCH_PAD_Y,
    padding_x=5,
    padding_y=2,
):
    """
    Given a template match (x0, y0) and template size (tw, th),
    look around it locally to find the full plot rectangle via contours.
    Returns (cx0, cy0, cx1, cy1) in full-image coordinates, or None if failed.
    """

    # region around the template to search for the full plot box
    rx0 = max(0, x0 - search_pad_x)
    ry0 = max(0, y0 - search_pad_y)
    rx1 = min(img_gray.shape[1], x0 + tw + search_pad_x)
    ry1 = min(img_gray.shape[0], y0 + th + search_pad_y)

    roi_gray = img_gray[ry0:ry1, rx0:rx1]

    # binarize (plots usually have a white background + dark frame/axes)
    _, binary = cv2.threshold(
        roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # invert (often works better for detecting dark frames/axes)
    inv = 255 - binary

    contours, _ = cv2.findContours(
        inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    best_rect = None
    best_area = 0

    # template point in ROI coordinates
    templ_center_x = x0 + tw // 2 - rx0
    templ_center_y = y0 + th // 2 - ry0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < MIN_CONTOUR_AREA:  # avoid tiny bits; tune if needed
            continue

        # does this rect contain the template center?
        if not (x <= templ_center_x <= x + w and y <= templ_center_y <= y + h):
            continue

        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)

    if best_rect is None:
        return None

    x, y, w, h = best_rect

    # map back to full-image coords and add gentle padding
    cx0 = max(0, rx0 + x - padding_x)
    cy0 = max(0, ry0 + y - padding_y)
    cx1 = min(img_gray.shape[1], rx0 + x + w + padding_x)
    cy1 = min(img_gray.shape[0], ry0 + y + h + padding_y)

    return cx0, cy0, cx1, cy1


# === LOAD TEMPLATE ===
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(f"Could not load template image at: {template_path}")

th, tw = template.shape[:2]

# === MAIN ===
doc = fitz.open(pdf_path)
global_plot_counter = 0

for page_idx, page in enumerate(doc):
    print(f"\n=== Page {page_idx + 1} ===")
    img_color = render_page_to_image(page, dpi=DPI)
    h, w = img_color.shape[:2]
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # template matching on page
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    matches = []  # list of (x, y, score)
    while True:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val < MATCH_THRESHOLD:
            print(
                f"Stopping: best remaining score {max_val:.3f} "
                f"below threshold {MATCH_THRESHOLD}."
            )
            break

        x0, y0 = max_loc
        matches.append((x0, y0, max_val))
        print(f"  Found match at ({x0}, {y0}) with score {max_val:.3f}")

        # suppress a region around this match
        sup_w = int(tw * SUPPRESS_SCALE_X)
        sup_h = int(th * SUPPRESS_SCALE_Y)
        sx0 = max(0, x0 - sup_w // 2)
        sy0 = max(0, y0 - sup_h // 2)
        sx1 = min(round(res.shape[1]), x0 + tw + sup_w // 2)
        sy1 = min(round(res.shape[0]), y0 + th + sup_h // 2)

        res[sy0:sy1, sx0:sx1] = 0.0

    print(f"Total matches on page {page_idx + 1}: {len(matches)}")

    # === GROUP MATCHES INTO ROWS BASED ON Y COORDINATE ===
    # each match is (x, y, score)
    if not matches:
        continue

    ROW_TOL = th * 0.6  # vertical tolerance: same row if y close within this

    matches_sorted_by_y = sorted(matches, key=lambda m: m[1])

    rows = []  # list of {'y_ref': float, 'items': [(x,y,score), ...]}

    for (x, y, score) in matches_sorted_by_y:
        placed = False
        for row in rows:
            if abs(y - row["y_ref"]) <= ROW_TOL:
                row["items"].append((x, y, score))
                placed = True
                break
        if not placed:
            rows.append({"y_ref": y, "items": [(x, y, score)]})

    # sort rows top to bottom
    rows.sort(key=lambda r: r["y_ref"])

    print(f"  Grouped into {len(rows)} row(s). Sizes:", [len(r["items"]) for r in rows])

    # === CROP & SAVE IN GRID ORDER (row, col) ===
    page_plot_count = 0

    for row_idx, row in enumerate(rows):
        # sort this row's plots left to right
        row_items = sorted(row["items"], key=lambda m: m[0])

        for col_idx, (x0, y0, score) in enumerate(row_items):
            coords = find_tight_plot_crop(
                img_gray, img_color, x0, y0, tw, th
            )
            if coords is None:
                # fallback: simple template-based box
                cx0 = max(0, x0 - 20)
                cy0 = max(0, y0 - 20)
                cx1 = min(w, x0 + tw + 20)
                cy1 = min(h, y0 + th + 20)
            else:
                cx0, cy0, cx1, cy1 = coords
            
            

            crop = img_color[cy0:cy1, cx0:cx1]
            print(crop.shape)
            if crop.size == 0:
                print(
                    f"  Skipping empty crop for row {row_idx}, col {col_idx}"
                )
                continue

            page_plot_count += 1
            global_plot_counter += 1

            # 0-based indices
            out_path = output_dir / (
                f"page{page_idx + 1:02d}_r{row_idx}_c{col_idx}.png"
            )

            # If you want 1-based indices instead, use this:
            # out_path = output_dir / (
            #     f"page{page_idx + 1:02d}_r{row_idx+1}_c{col_idx+1}.png"
            # )

            ok = cv2.imwrite(str(out_path), crop)
            print(
                f"  Saved plot r{row_idx},c{col_idx} (global {global_plot_counter}) "
                f"score={score:.3f} -> {out_path} (success={ok})"
            )

print(f"\nDone. Total plots across all pages: {global_plot_counter}")
