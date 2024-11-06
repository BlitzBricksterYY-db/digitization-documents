# Databricks notebook source
!pip install pytesseract
!pip install Pillow
!pip install pdf2image
!pip install poppler-utils

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sh
# MAGIC # Update package list and install Poppler
# MAGIC sudo apt-get update
# MAGIC sudo apt-get install -y poppler-utils

# COMMAND ----------

# MAGIC %sh
# MAGIC pdftoppm -v

# COMMAND ----------

# MAGIC %sh
# MAGIC # Update package list and install Tesseract OCR
# MAGIC sudo apt-get update
# MAGIC sudo apt-get install -y tesseract-ocr

# COMMAND ----------

# MAGIC %sh
# MAGIC tesseract --version

# COMMAND ----------

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# COMMAND ----------

images = convert_from_path('W291WIE02499.pdf')

# COMMAND ----------

# Prepare list to hold all data
results_word = []

# Process each page
for page_number, page_image in enumerate(images, start=1):
    # Convert page image to grayscale (optional but can improve OCR results)
    page_image = page_image.convert('L')
    
    # Get OCR data
    ocr_data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)
    
    # Extract relevant data into JSON-like structure
    for i in range(len(ocr_data['level'])):
        block = {
            "BlockType": "WORD",
            "Text": ocr_data['text'][i],
            "Confidence": float(ocr_data['conf'][i]) if ocr_data['conf'][i] != -1 else None,
            "Geometry": {
                "BoundingBox": {
                    "Left": float(ocr_data['left'][i] / page_image.width),
                    "Top": float(ocr_data['top'][i] / page_image.height),
                    "Width": float(ocr_data['width'][i] / page_image.width),
                    "Height": float(ocr_data['height'][i] / page_image.height),
                }
            },
            "Id": i + 1
        }
        
        # Only include blocks with text
        if block["Text"].strip():
            results_word.append(block)

# COMMAND ----------

results_word

# COMMAND ----------

# Prepare list to hold all data
results_line = []

# Process each page
for page_number, page_image in enumerate(images, start=1):
    # Convert page image to grayscale (optional but can improve OCR results)
    page_image = page_image.convert('L')
    
    # Get OCR data with line-level information
    ocr_data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)
    
    # Initialize variables to store current line, line confidence, and track line numbers
    current_line = ""
    last_line_num = -1
    line_confidences = []
    
    for i in range(len(ocr_data['level'])):
        line_num = ocr_data['line_num'][i]
        
        # If confidence is valid (not -1), add it to the line confidence list
        if ocr_data['conf'][i] != '-1':
            line_confidences.append(float(ocr_data['conf'][i]))
        
        # Check if the line number has changed, indicating a new line
        if line_num != last_line_num and current_line:
            # Calculate the average confidence for the current line
            avg_confidence = sum(line_confidences) / len(line_confidences) if line_confidences else None
            
            # Save the current line as a block
            results_line.append({
                "BlockType": "LINE",
                "Text": current_line.strip(),
                "Confidence": avg_confidence,
                "Geometry": {
                    "BoundingBox": {
                        "Left": float(ocr_data['left'][i] / page_image.width),
                        "Top": float(ocr_data['top'][i] / page_image.height),
                        "Width": float(ocr_data['width'][i] / page_image.width),
                        "Height": float(ocr_data['height'][i] / page_image.height),
                    }
                },
                "Id": i + 1
            })
            # Reset the line text and line confidences
            current_line = ""
            line_confidences = []
        
        # Append current word to the line
        if ocr_data['text'][i].strip():
            current_line += ocr_data['text'][i] + " "
        
        last_line_num = line_num

    # Append the last line on the page if needed
    if current_line.strip():
        avg_confidence = sum(line_confidences) / len(line_confidences) if line_confidences else None
        results_line.append({
            "BlockType": "LINE",
            "Text": current_line.strip(),
            "Confidence": avg_confidence,
            "Geometry": {
                "BoundingBox": {
                    "Left": float(ocr_data['left'][i] / page_image.width),
                    "Top": float(ocr_data['top'][i] / page_image.height),
                    "Width": float(ocr_data['width'][i] / page_image.width),
                    "Height": float(ocr_data['height'][i] / page_image.height),
                }
            },
            "Id": len(results_line) + 1
        })

# COMMAND ----------

results_line

# COMMAND ----------

# Prepare list to hold all data
results_paragraph = []

# Process each page
for page_number, page_image in enumerate(images, start=1):
    # Convert page image to grayscale (optional but can improve OCR results)
    page_image = page_image.convert('L')
    
    # Get OCR data with block-level information
    ocr_data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)
    
    # Initialize variables to store current paragraph, block confidence, and track block numbers
    current_paragraph = ""
    last_block_num = -1
    block_confidences = []
    unique_id = 0  # Unique ID for each block
    
    for i in range(len(ocr_data['level'])):
        block_num = ocr_data['block_num'][i]
        
        # If confidence is valid (not -1), add it to the block confidence list
        if ocr_data['conf'][i] != '-1':
            block_confidences.append(float(ocr_data['conf'][i]))
        
        # Check if the block number has changed, indicating a new paragraph
        if block_num != last_block_num and current_paragraph:
            # Calculate the average confidence for the current block
            avg_confidence = sum(block_confidences) / len(block_confidences) if block_confidences else None
            
            # Save the current paragraph as a block
            results_paragraph.append({
                "BlockType": "PARAGRAPH",
                "Text": current_paragraph.strip(),
                "Confidence": avg_confidence,
                "PageNumber": page_number,
                "Geometry": {
                    "BoundingBox": {
                        "Left": float(ocr_data['left'][i] / page_image.width),
                        "Top": float(ocr_data['top'][i] / page_image.height),
                        "Width": float(ocr_data['width'][i] / page_image.width),
                        "Height": float(ocr_data['height'][i] / page_image.height),
                    }
                },
                "Id": unique_id
            })
            # Increment the unique ID for the next block
            unique_id += 1
            # Reset the paragraph text and block confidences
            current_paragraph = ""
            block_confidences = []
        
        # Append current word to the paragraph
        if ocr_data['text'][i].strip():
            current_paragraph += ocr_data['text'][i] + " "
        
        last_block_num = block_num

    # Append the last paragraph on the page if needed
    if current_paragraph.strip():
        avg_confidence = sum(block_confidences) / len(block_confidences) if block_confidences else None
        results_paragraph.append({
            "BlockType": "PARAGRAPH",
            "Text": current_paragraph.strip(),
            "Confidence": avg_confidence,
            "PageNumber": page_number,
            "Geometry": {
                "BoundingBox": {
                    "Left": float(ocr_data['left'][i] / page_image.width),
                    "Top": float(ocr_data['top'][i] / page_image.height),
                    "Width": float(ocr_data['width'][i] / page_image.width),
                    "Height": float(ocr_data['height'][i] / page_image.height),
                }
            },
            "Id": unique_id
        })
        unique_id += 1  # Increment for the last block

# COMMAND ----------

results_paragraph

# COMMAND ----------

# Print or save results as JSON
import json
with open('ocr_output.json', 'w') as json_file:
    json.dump({"Blocks": results_paragraph}, json_file, indent=4)

print("OCR extraction complete. Check 'ocr_output.json' for results.")

# COMMAND ----------

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# Load OCR results from the JSON file
with open('ocr_output.json', 'r') as json_file:
    ocr_results = json.load(json_file)

# Function to display the image with the bounding box of the selected paragraph
def view_paragraph_with_bounding_box(pdf_path, page_number, paragraph_id):
    # Convert the specified PDF page to an image
    pages = convert_from_path(pdf_path, dpi=300)
    
    if page_number > len(pages) or page_number < 1:
        print("Invalid page number.")
        return
    
    # Get the image for the specified page
    page_image = pages[page_number - 1]

    # Find the specified paragraph by ID and page number
    paragraph = None
    for block in ocr_results['Blocks']:
        if block['Id'] == paragraph_id and block['BlockType'] == 'PARAGRAPH' and block['PageNumber'] == page_number:
            paragraph = block
            break
    
    if not paragraph:
        print("Paragraph ID not found.")
        return
    
    # Display the image with the bounding box
    fig, ax = plt.subplots(1, figsize=(10, 15))
    ax.imshow(page_image)
    
    # Draw the bounding box
    bbox = paragraph['Geometry']['BoundingBox']
    left = bbox['Left'] * page_image.width
    top = bbox['Top'] * page_image.height
    width = bbox['Width'] * page_image.width
    height = bbox['Height'] * page_image.height
    
    rect = patches.Rectangle(
        (left, top), width, height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Display paragraph text
    plt.title(f"Page Number: {page_number} | Paragraph ID: {paragraph_id}\nText: {paragraph['Text']}")
    plt.show()

# COMMAND ----------

# Example usage
pdf_path = 'W291WIE02499.pdf'  # Replace with your PDF file path
page_number = int(input("Enter the page number: "))
paragraph_id = int(input("Enter the paragraph ID: "))

view_paragraph_with_bounding_box(pdf_path, page_number, paragraph_id)

# COMMAND ----------

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

# Load OCR results from the JSON file
with open('ocr_output.json', 'r') as json_file:
    ocr_results = json.load(json_file)

# Function to display all bounding boxes for a page
def view_all_bounding_boxes(pdf_path, page_number):
    # Convert the specified PDF page to an image
    pages = convert_from_path(pdf_path, dpi=300)
    
    if page_number > len(pages) or page_number < 1:
        print("Invalid page number.")
        return
    
    # Get the image for the specified page
    page_image = pages[page_number - 1]

    # Display the image
    fig, ax = plt.subplots(1, figsize=(10, 15))
    ax.imshow(page_image)
    
    # Draw bounding boxes for all paragraphs on the specified page
    for block in ocr_results['Blocks']:
        if block['BlockType'] == 'PARAGRAPH' and block['PageNumber'] == page_number:  # Filter by page number
            bbox = block['Geometry']['BoundingBox']
            left = bbox['Left'] * page_image.width
            top = bbox['Top'] * page_image.height
            width = bbox['Width'] * page_image.width
            height = bbox['Height'] * page_image.height
            
            # Draw the bounding box
            rect = patches.Rectangle(
                (left, top), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Optionally, label the bounding box with the ID
            ax.text(left, top, f"ID: {block['Id']}", color='blue', fontsize=8)

    plt.title(f"Bounding Boxes on Page {page_number}")
    plt.show()

# COMMAND ----------

# Example usage
pdf_path = 'W291WIE02499.pdf'  # Replace with your PDF file path
page_number = int(input("Enter the page number: "))

view_all_bounding_boxes(pdf_path, page_number)
