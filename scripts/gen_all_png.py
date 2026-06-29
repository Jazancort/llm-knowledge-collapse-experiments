"""Re-generate all figures as PNG 300dpi for LaTeX compatibility."""
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
FIG_DIR = SCRIPTS_DIR.parent / "paper" / "figs"

scripts = sorted(SCRIPTS_DIR.glob("gen_fig*.py"))
for s in scripts:
    code = s.read_text(encoding="utf-8")
    # Add png output after existing saves
    code_mod = code.replace(
        'fig.savefig(FIG_DIR / "fig',
        'fig.savefig(FIG_DIR / "fig'
    )
    # Just replace .pdf with .png in the first savefig call
    lines = code.splitlines()
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "savefig" in line and ".pdf" in line:
            # Add a PNG save right after
            png_line = line.replace(".pdf", ".png")
            if "dpi=300" not in png_line:
                png_line = png_line.replace("bbox_inches=\"tight\"", "bbox_inches=\"tight\", dpi=300")
            new_lines.append(png_line)
    
    tmp = SCRIPTS_DIR / "_tmp_gen_png.py"
    tmp.write_text("\n".join(new_lines), encoding="utf-8")
    result = subprocess.run([sys.executable, str(tmp)], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{s.name} -> PNG OK")
    else:
        print(f"{s.name} FAILED: {result.stderr[:300]}")
    tmp.unlink()

# Verify PNGs exist
pngs = sorted(FIG_DIR.glob("fig*.png"))
print(f"\nGenerated {len(pngs)} PNGs:")
for p in pngs:
    print(f"  {p.name} ({p.stat().st_size // 1024}KB)")
