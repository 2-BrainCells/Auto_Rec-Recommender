from PyPDF2 import PdfReader
reader = PdfReader("1-s2.0-S0957417424006043-main-2.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
with open("extracted_paper.txt", "w", encoding="utf-8") as f:
    f.write(text)
