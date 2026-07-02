"""Capture fig1_overview.html as high-resolution PNG for the paper."""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
HTML_FILE = SCRIPT_DIR / "fig1_overview.html"
OUTPUT_DIR = SCRIPT_DIR.parent / "paper" / "figs"
OUTPUT_PNG = OUTPUT_DIR / "fig1_overview.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use Chrome/Edge headless to capture at high DPI
# Try multiple browser paths
browsers = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

browser = None
for b in browsers:
    if Path(b).exists():
        browser = b
        break

if not browser:
    print("ERROR: No Chrome or Edge found. Install Chrome or capture manually.")
    print(f"HTML file: {HTML_FILE}")
    sys.exit(1)

# Capture at 2x scale for high DPI
cmd = [
    browser,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    f"--screenshot={OUTPUT_PNG}",
    "--window-size=1800,580",
    "--force-device-scale-factor=2",
    "--hide-scrollbars",
    f"file:///{HTML_FILE.as_posix()}",
]

print(f"Capturing {HTML_FILE.name} -> {OUTPUT_PNG.name}")
print(f"Browser: {Path(browser).name}")

result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

if result.returncode == 0 and OUTPUT_PNG.exists():
    # Auto-crop whitespace from bottom
    try:
        from PIL import Image, ImageChops
        img = Image.open(OUTPUT_PNG)
        # Create a white background to compare
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            # Add small margin
            cropped = img.crop((0, 0, img.width, min(bbox[3] + 20, img.height)))
            cropped.save(OUTPUT_PNG)
            print(f"Cropped: {img.height}px -> {cropped.height}px")
    except ImportError:
        print("Pillow not installed, skipping auto-crop")

    size_kb = OUTPUT_PNG.stat().st_size / 1024
    print(f"Success! Output: {OUTPUT_PNG} ({size_kb:.0f} KB)")
else:
    print(f"Warning: return code {result.returncode}")
    if result.stderr:
        print(f"stderr: {result.stderr[:500]}")
    if OUTPUT_PNG.exists():
        size_kb = OUTPUT_PNG.stat().st_size / 1024
        print(f"File exists anyway: {OUTPUT_PNG} ({size_kb:.0f} KB)")
    else:
        print("File not created. Try opening the HTML manually in a browser and using a screenshot tool.")
