"""Extract annotations from Carlos v2 review PDF."""
import fitz

doc = fitz.open(r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\Review\Carlos\v2\Knowledge Degradation in LLMs - Julio Azancort (1).pdf")
annotations = []

for page_num in range(len(doc)):
    page = doc[page_num]
    for annot in page.annots() or []:
        info = annot.info
        content = info.get("content", "").strip()
        subject = info.get("subject", "")
        annot_type = annot.type[1] if annot.type else ""
        
        # Get highlighted text if available
        highlight_text = ""
        if annot.type[0] == 8:  # Highlight
            quads = annot.vertices
            if quads:
                rects = [fitz.Quad(quads[i:i+4]).rect for i in range(0, len(quads), 4)]
                for r in rects:
                    highlight_text += page.get_text("text", clip=r).strip() + " "
        
        annotations.append({
            "page": page_num + 1,
            "type": annot_type,
            "content": content,
            "highlight": highlight_text.strip(),
            "subject": subject,
        })

print(f"Total annotations: {len(annotations)}")
print("---")

output_lines = []
output_lines.append(f"Total annotations: {len(annotations)}\n")

for i, a in enumerate(annotations, 1):
    page = a["page"]
    atype = a["type"]
    output_lines.append(f"#{i} [Page {page}] ({atype})")
    if a["highlight"]:
        hl = a["highlight"][:200]
        output_lines.append(f"  HIGHLIGHT: {hl}")
    if a["content"]:
        output_lines.append(f"  COMMENT: {a['content']}")
    output_lines.append("")

out_path = r"G:\Lab\Labcity\LLM\Artigo\Paradoxo - springer\Paradoxo\Review\Carlos\v2\annotations_extracted.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"Saved to {out_path}")
