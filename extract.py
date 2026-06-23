import fitz

# Extract Fee_structure.pdf → fees.txt
pdf1 = fitz.open("data/Fee_structure.pdf")
text1 = ""
for page in pdf1:
    text1 += page.get_text()
with open("data/fees.txt", "w", encoding="utf-8") as f:
    f.write(text1)
print("✅ fees.txt filled from Fee_structure.pdf")

# Extract Prospectus → admission.txt
pdf2 = fitz.open("data/Prospectus_2026-27_@_24_March.pdf")
text2 = ""
for page in pdf2:
    text2 += page.get_text()
with open("data/admission.txt", "w", encoding="utf-8") as f:
    f.write(text2)
print("✅ admission.txt filled from Prospectus PDF")