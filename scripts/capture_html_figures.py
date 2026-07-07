"""Recapture HTML figures at 4x device scale for print quality (450+ DPI)."""
from playwright.sync_api import sync_playwright
from pathlib import Path

FIGS = Path(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v2\figs")

figures = [
    ("fig1_overview.html", "fig1_overview.png", 1400),
    ("fig_regime_examples.html", "fig_regime_examples.png", 1400),
]

with sync_playwright() as p:
    browser = p.chromium.launch()
    for html_file, png_file, width in figures:
        html_path = FIGS / html_file
        if not html_path.exists():
            print(f"SKIP {html_file} (not found)")
            continue
        ctx = browser.new_context(device_scale_factor=4, viewport={"width": width, "height": 100})
        page = ctx.new_page()
        page.goto(html_path.as_uri())
        page.screenshot(path=str(FIGS / png_file), full_page=True)
        ctx.close()
        from PIL import Image
        img = Image.open(FIGS / png_file)
        print(f"{png_file}: {img.size[0]}x{img.size[1]} px ({img.size[0]/width:.0f}x scale)")
    browser.close()
