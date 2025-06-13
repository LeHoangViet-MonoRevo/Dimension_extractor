import glob
import os

from pdf2image import convert_from_path

# Set input and output directories
input_dir = "dataset"
output_dir = "dataset_converted"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all PDF files
pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
print(f"Num pdf: {len(pdf_files)}")

# Convert first page of each PDF
for pdf_path in pdf_files:
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{pdf_name}.png")

    try:
        # Convert only the first page
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        images[0].save(output_path, "PNG")
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to convert {pdf_path}: {e}")
