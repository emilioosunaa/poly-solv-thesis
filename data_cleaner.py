from docling.document_converter import DocumentConverter

# 1. Path to your PDF file
source = "Polymer-Solvent Interaction Parameter.pdf"

# 2. Initialize the converter
converter = DocumentConverter()

# 3. Convert the PDF into Docling’s document model
result = converter.convert(source)

# 4. Export the model to Markdown
markdown = result.document.export_to_markdown()

# 5. Write out to a file
with open("polymer_solvent_interaction_parameter.md", "w", encoding="utf-8") as f:
    f.write(markdown)

print("✅ Markdown has been written to polymer_solvent_interaction_parameter.md")
