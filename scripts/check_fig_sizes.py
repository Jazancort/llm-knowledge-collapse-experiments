from PIL import Image
from pathlib import Path

figs_dir = Path(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\llm-knowledge-collapse (paper)\v3\figs")

for f in sorted(figs_dir.glob("*.png")):
    img = Image.open(f)
    dpi = img.info.get("dpi", (72, 72))
    w_in = img.size[0] / dpi[0]
    h_in = img.size[1] / dpi[1]
    print(f"{f.name:40s} {img.size[0]:5d}x{img.size[1]:5d}  dpi={dpi[0]:.0f}  {w_in:.1f}x{h_in:.1f} in  ratio={img.size[1]/img.size[0]:.2f}")
