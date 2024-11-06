# Databricks notebook source
# MAGIC %md
# MAGIC # README
# MAGIC
# MAGIC Test PDF extraction solutions for Centene

# COMMAND ----------

# MAGIC %md
# MAGIC ref: https://nanonets.com/blog/ocr-with-tesseract/

# COMMAND ----------

# MAGIC %md
# MAGIC The --oem option in Tesseract specifies the OCR Engine Mode to use. Here are the available options:
# MAGIC
# MAGIC 0: Legacy engine only.
# MAGIC 1: Neural nets LSTM engine only.
# MAGIC 2: Legacy + LSTM engines.
# MAGIC 3: Default, based on what is available.
# MAGIC Recommendations:
# MAGIC For most modern applications, using the LSTM engine (--oem 1) is recommended as it generally provides better accuracy.
# MAGIC If you need compatibility with older models or specific features of the legacy engine, you might use --oem 0.
# MAGIC For a combination of both engines, you can use --oem 2, though this might be slower.
# MAGIC Using the default option (--oem 3) is a safe choice if you are unsure, as it will select the best available engine.
# MAGIC Example Usage:
# MAGIC 12
# MAGIC custom_config = r'--oem 1 --psm 6'
# MAGIC data = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
# MAGIC This configuration uses the LSTM engine and treats the image as a single uniform block of text lines.

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

import os

# landing_zone = "/Volumes/yyang/centene_testing/pdf"
print("Storage location for landing zone is: {}".format(landing_zone))
print("Content: {}".format(dbutils.fs.ls(landing_zone)))
print("Content: {}".format(os.listdir(landing_zone,  )))

# COMMAND ----------

# MAGIC %md
# MAGIC # Split the whold pdf into pages and Conquer strategy

# COMMAND ----------

# MAGIC %md
# MAGIC ## Separate pages helper functions
# MAGIC We would like to separate pages by complexity of parsing. While some pages may contain plain text that will be extracted as-is, others may include tables that could benefit from a post processing engine such as AWS textract. For that purpose, we split our various PDF as multiple pages documents that we store individually on our cloud storage together with a unique identifier (will be useful for our post processing logic).

# COMMAND ----------

from pypdf import PdfReader
from pypdf import PdfWriter
from io import BytesIO


def convert_page_pdf(page):
    """
    Convert a given page object into its own PDF
    :param pageObject page: the extracted page object
    """
    writer = PdfWriter()
    writer.add_page(page)
    tmp = BytesIO()
    writer.write(tmp)
    return tmp.getvalue()
  
  
def split_pages(content):
    """
    For each document, we extract each individual page, converting into a single document
    This process is key to apply downstream business logic dynamically depending on the page content
    :param binary content: the original PDF document as binary
    """
    pages = []
    reader = PdfReader(BytesIO(content))
    number_of_pages = len(reader.pages)
    for page_number in range(0, number_of_pages):
        page = reader.pages[page_number] # retrieve specific page
        page_text = page.extract_text() # extract plain text content
        page_content = convert_page_pdf(page) # each page will become its own PDF
        pages.append(page_content)
    return pages

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split pdf into pages and store into individual pdfs

# COMMAND ----------

from pyspark.sql.functions import input_file_name

# Read PDF files as binary
binary_df = spark.read.format('binaryFile').load(landing_zone)
# binary_df = binary_df.withColumn("path", input_file_name())

# COMMAND ----------

binary_df.display()

# COMMAND ----------

# DBTITLE 1,PDF Content Handling and Page Splitter Script
import uuid

# Display the content of the PDFs
# Collect the data from the DataFrame
binary_data = binary_df.collect()

# Display the content of the PDFs
for row in binary_data:
    path = row['path']
    content = row['content']
    print(f"Content of {path}:")
    print(content[:100])
            
    try:
      # generate a unique identifier and a unique path where files will be stored
      doc_id = uuid.uuid4().hex
      dir = '/{}/{}/pages'.format(landing_zone, doc_id)
      os.makedirs(dir, exist_ok=True)
      # split PDF into individual pages
      pages = split_pages(content)

      # write each page individually to storage
      for j, page_content in enumerate(pages):
          with open('{}/{}.pdf'.format(dir, j + 1), 'wb') as f:
              f.write(page_content)

      print('Split into pages for [{}]'.format(path))
    except:
        print('Failed to split report for [{}]'.format(path))
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC # Extract from splits

# COMMAND ----------

# DBTITLE 1,Displaying and Managing Storage Location in Python
import os

landing_zone = "/Volumes/yyang/centene_testing/pdf"
print("Storage location for landing zone is: {}".format(landing_zone))
print("Content: {}".format(dbutils.fs.ls(landing_zone)))
print("Content: {}".format(os.listdir(landing_zone,  )))
landing_zone_fs = '{}/**/pages'.format(landing_zone)
print("landing_zone_fs is {}".format(landing_zone_fs))
# 
bad_records_path = "{}/badRecordsPath".format(landing_zone)
dbutils.fs.mkdirs(bad_records_path)
print("bad_records_path: {}".format(bad_records_path))

# COMMAND ----------

try: 
  tika_df = spark.read.format('tika').option("badRecordsPath", bad_records_path).load(landing_zone_fs)
  display(tika_df)
except Exception as e:
    print(f"Failed to read files using Tika format: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Milestone Conclusion
# MAGIC The digital pdf were extracted without a problem, but image-based pdf failed and diverted to the badRecordsPath.

# COMMAND ----------

# MAGIC %md
# MAGIC # Extracts from splits enabling OCR

# COMMAND ----------

# we know it is sure to have this folder for failed pdf - image based. 
landing_zone_fs = landing_zone_fs.replace("**", "39b3cee5d3554fb090bd585005288313")
print("landing_zone_fs is {}".format(landing_zone_fs))

# COMMAND ----------


# 39b3cee5d3554fb090bd585005288313
try: 
  tika_df = spark.read.format('tika').option("badRecordsPath", bad_records_path) \
  .option("pathGlobFilter", "*.pdf") \
  .option("ocr", "true") \
  .load(landing_zone_fs)
  display(tika_df)
except Exception as e:
    print(f"Failed to read files using Tika format: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Using pytesseract

# COMMAND ----------

# MAGIC %md
# MAGIC Tesseract 4.00 includes a new neural network subsystem configured as a text line recognizer. It has its origins in OCRopus' Python-based LSTM implementation but has been redesigned for Tesseract in C++. The neural network system in Tesseract pre-dates TensorFlow but is compatible with it, as there is a network description language called Variable Graph Specification Language (VGSL), that is also available for TensorFlow.
# MAGIC
# MAGIC To recognize an image containing a single character, we typically use a Convolutional Neural Network (CNN). Text of arbitrary length is a sequence of characters, and such problems are solved using RNNs and LSTM is a popular form of RNN. Read this post to learn more about LSTM.
# MAGIC
# MAGIC

# COMMAND ----------

dbutils.fs.ls(landing_zone)

# COMMAND ----------

# MAGIC %md
# MAGIC Prompt:
# MAGIC
# MAGIC Given a multi-page scanned pdf file, please use pytesseract to parse it and return a JSON format output with multiple identified blocks. Each block has this format:
# MAGIC   ```{"BlockType": "WORD",
# MAGIC   "Text": "mhs health",
# MAGIC   "Confidence": 0.9963087366826908,
# MAGIC   "Geometry": {
# MAGIC     "BoundingBox": {
# MAGIC     "Left": 0.07058823529411765,
# MAGIC     "Top": 0.05545454545454546,
# MAGIC     "Width": 0.14705882352941177,
# MAGIC     "Height": 0.025454545454545455}
# MAGIC     },
# MAGIC   "Id": 1}```

# COMMAND ----------

import json
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

images = convert_from_path(pdf_path)


# COMMAND ----------

len(images)

# COMMAND ----------

for image in images:
  data = pytesseract.image_to_data(image, output_type=Output.DICT)

# COMMAND ----------

str(data)

# COMMAND ----------

data.items()

# COMMAND ----------

{f'len of {k}': len(v) for k, v in data.items()}

# COMMAND ----------

data.keys()

# COMMAND ----------

# DBTITLE 1,word level
import json
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

# Function to convert PDF to images and parse text using PyTesseract
def parse_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    blocks = []
    block_id = 1
    
    for image in images:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                block = {
                    "BlockType": "WORD",
                    "Text": data['text'][i],
                    "Confidence": data['conf'][i],
                    "Geometry": {
                        "BoundingBox": {
                            "Left": data['left'][i] / image.width,
                            "Top": data['top'][i] / image.height,
                            "Width": data['width'][i] / image.width,
                            "Height": data['height'][i] / image.height
                        }
                    },
                    "Id": block_id
                }
                blocks.append(block)
                block_id += 1
                
    return json.dumps(blocks, indent=2)

# Specify the path to your PDF file
pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

# Parse the PDF and get JSON output
json_output = parse_pdf(pdf_path)
print(json_output)

# COMMAND ----------

# DBTITLE 1,psm 6 doesn't do anything regarding lines
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Function to convert PDF to images and parse text using PyTesseract
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    block_id = 1
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR format for OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        custom_config = r'--psm 6'
        data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                block = {
                    "BlockType": "LINE",
                    "Text": data['text'][i],
                    "Confidence": data['conf'][i],
                    "Geometry": {
                        "BoundingBox": {
                            "Left": data['left'][i] / img.shape[1],
                            "Top": data['top'][i] / img.shape[0],
                            "Width": data['width'][i] / img.shape[1],
                            "Height": data['height'][i] / img.shape[0]
                        }
                    },
                    "Id": block_id
                }
                blocks.append(block)
                block_id += 1
                
    return json.dumps(blocks, indent=2)

# Specify the path to your PDF file
pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

# Parse the PDF and get JSON output
json_output = parse_pdf(pdf_path)
print(json_output)

# COMMAND ----------

# DBTITLE 1,line level v1
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Function to convert PDF to images and parse text using PyTesseract
# notice block_id is incremented from 1 until reach the last block of last page of the pdf.
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    block_id = 1
    #"Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR format for OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # custom_config = r'--psm 6'
        # data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)


        current_line = ""
        current_bbox = [float('inf'), float('inf'), 0, 0]  # left, top, right, bottom
        confidence = []

        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                left = data['left'][i]
                top = data['top'][i]
                width = data['width'][i]
                height = data['height'][i]
                
                # new block triggering condition
                if current_line and top > current_bbox[1] + current_bbox[3]:
                    block = {
                        "BlockType": "LINE",
                        "Text": current_line.strip(),
                        "Confidence": np.nanmean(confidence),
                            # sum(data['conf'][j] for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]) / len([j for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]]),
                        "Geometry": {
                            "BoundingBox": {
                                "Left": current_bbox[0] / img.shape[1],
                                "Top": current_bbox[1] / img.shape[0],
                                "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                                "Height": current_bbox[3] / img.shape[0]
                            }
                        },
                        "Id": block_id,
                        "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
                    }
                    blocks.append(block)
                    block_id += 1
                    # reset below counters and containers
                    current_line = ""
                    current_bbox = [float('inf'), float('inf'), 0, 0]
                    confidence = []
                
                current_line += data['text'][i] + " "
                confidence.append(data['conf'][i]) 
                current_bbox[0] = min(current_bbox[0], left)
                current_bbox[1] = min(current_bbox[1], top)
                current_bbox[2] = max(current_bbox[2], left + width)
                current_bbox[3] = max(current_bbox[3], height)
        
        # add the last line into the last new block since it is skipped in the loop above to form a new block.
        if current_line:
            block = {
                "BlockType": "LINE",
                "Text": current_line.strip(),
                "Confidence": np.nanmean(confidence),
                    #sum(data['conf'][j] for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]) / len([j for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]]),
                "Geometry": {
                    "BoundingBox": {
                        "Left": current_bbox[0] / img.shape[1],
                        "Top": current_bbox[1] / img.shape[0],
                        "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                        "Height": current_bbox[3] / img.shape[0]
                    }
                },
                "Id": block_id,
                "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
            }
            blocks.append(block)
            block_id += 1
                
    return json.dumps(blocks, indent=2)

# Specify the path to your PDF file
pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

# Parse the PDF and get JSON output
json_output = parse_pdf(pdf_path)
print(json_output)

# COMMAND ----------

# DBTITLE 1,line level v1.5 add gray scale
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Function to convert PDF to images and parse text using PyTesseract
# notice block_id is incremented from 1 until reach the last block of last page of the pdf.
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    block_id = 1
    #"Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR format for OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # custom_config = r'--psm 6'
        # data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, output_type=Output.DICT)


        current_line = ""
        current_bbox = [float('inf'), float('inf'), 0, 0]  # left, top, right, bottom
        confidence = []

        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                left = data['left'][i]
                top = data['top'][i]
                width = data['width'][i]
                height = data['height'][i]
                
                # new block triggering condition
                if current_line and top > current_bbox[1] + current_bbox[3]:
                    block = {
                        "BlockType": "LINE",
                        "Text": current_line.strip(),
                        "Confidence": np.nanmean(confidence),
                            # sum(data['conf'][j] for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]) / len([j for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]]),
                        "Geometry": {
                            "BoundingBox": {
                                "Left": current_bbox[0] / img.shape[1],
                                "Top": current_bbox[1] / img.shape[0],
                                "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                                "Height": current_bbox[3] / img.shape[0]
                            }
                        },
                        "Id": block_id,
                        "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
                    }
                    blocks.append(block)
                    block_id += 1
                    # reset below counters and containers
                    current_line = ""
                    current_bbox = [float('inf'), float('inf'), 0, 0]
                    confidence = []
                
                current_line += data['text'][i] + " "
                confidence.append(data['conf'][i]) 
                current_bbox[0] = min(current_bbox[0], left)
                current_bbox[1] = min(current_bbox[1], top)
                current_bbox[2] = max(current_bbox[2], left + width)
                current_bbox[3] = max(current_bbox[3], height)
        
        # add the last line into the last new block since it is skipped in the loop above to form a new block.
        if current_line:
            block = {
                "BlockType": "LINE",
                "Text": current_line.strip(),
                "Confidence": np.nanmean(confidence),
                    #sum(data['conf'][j] for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]) / len([j for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]]),
                "Geometry": {
                    "BoundingBox": {
                        "Left": current_bbox[0] / img.shape[1],
                        "Top": current_bbox[1] / img.shape[0],
                        "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                        "Height": current_bbox[3] / img.shape[0]
                    }
                },
                "Id": block_id,
                "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
            }
            blocks.append(block)
            block_id += 1
                
    return json.dumps(blocks, indent=2)

# Specify the path to your PDF file
pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

# Parse the PDF and get JSON output
json_output = parse_pdf(pdf_path)
print(json_output)

# COMMAND ----------

# MAGIC %md
# MAGIC The --oem option in Tesseract specifies the OCR Engine Mode to use. Here are the available options:
# MAGIC
# MAGIC 0: Legacy engine only.  
# MAGIC 1: Neural nets LSTM engine only.  
# MAGIC 2: Legacy + LSTM engines.  
# MAGIC 3: Default, based on what is available.  
# MAGIC __Recommendations:__
# MAGIC
# MAGIC + For most modern applications, using the LSTM engine (--oem 1) is recommended as it generally provides better accuracy.
# MAGIC + If you need compatibility with older models or specific features of the legacy engine, you might use --oem 0.
# MAGIC + For a combination of both engines, you can use --oem 2, though this might be slower.
# MAGIC + Using the default option (--oem 3) is a safe choice if you are unsure, as it will select the best available engine.

# COMMAND ----------

# MAGIC %md
# MAGIC prompt: preprocess with greyscale, denoise and increase the contrast level of the text on the page to make the background is mostly full white and the text mostly full black. Remove any other preprocessing of image.

# COMMAND ----------

# DBTITLE 1,line level v2 w preprocessing
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Function to convert PDF to images and parse text using PyTesseract
# notice block_id is incremented from 1 until reach the last block of last page of the pdf.
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    block_id = 1
    #"Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR format for OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Image preprocessing: grayscale, denoise, and increase contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        custom_config = r'--oem 1'  # Use LSTM OCR Engine
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)

        current_line = ""
        current_bbox = [float('inf'), float('inf'), 0, 0]  # left, top, right, bottom
        confidence = []

        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                left = data['left'][i]
                top = data['top'][i]
                width = data['width'][i]
                height = data['height'][i]
                
                # new block triggering condition
                if current_line and top > current_bbox[1] + current_bbox[3]:
                    block = {
                        "BlockType": "LINE",
                        "Text": current_line.strip(),
                        "Confidence": np.nanmean(confidence),
                        "Geometry": {
                            "BoundingBox": {
                                "Left": current_bbox[0] / img.shape[1],
                                "Top": current_bbox[1] / img.shape[0],
                                "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                                "Height": current_bbox[3] / img.shape[0]
                            }
                        },
                        "Id": block_id,
                        "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
                    }
                    blocks.append(block)
                    block_id += 1
                    # reset below counters and containers
                    current_line = ""
                    current_bbox = [float('inf'), float('inf'), 0, 0]
                    confidence = []
                
                current_line += data['text'][i] + " "
                confidence.append(data['conf'][i]) 
                current_bbox[0] = min(current_bbox[0], left)
                current_bbox[1] = min(current_bbox[1], top)
                current_bbox[2] = max(current_bbox[2], left + width)
                current_bbox[3] = max(current_bbox[3], height)
        
        # add the last line into the last new block since it is skipped in the loop above to form a new block.
        if current_line:
            block = {
                "BlockType": "LINE",
                "Text": current_line.strip(),
                "Confidence": np.nanmean(confidence),
                "Geometry": {
                    "BoundingBox": {
                        "Left": current_bbox[0] / img.shape[1],
                        "Top": current_bbox[1] / img.shape[0],
                        "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                        "Height": current_bbox[3] / img.shape[0]
                    }
                },
                "Id": block_id,
                "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
            }
            blocks.append(block)
            block_id += 1
                
    return json.dumps(blocks, indent=2)

# Specify the path to your PDF file
pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

# Parse the PDF and get JSON output
json_output = parse_pdf(pdf_path)
print(json_output)

# COMMAND ----------

# DBTITLE 1,line level v3 w preprocessing
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Function to convert PDF to images and parse text using PyTesseract
# notice block_id is incremented from 1 until reach the last block of last page of the pdf.
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    block_id = 1
    #"Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR format for OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Image preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        custom_config = r'--oem 1'  # Use LSTM OCR Engine
        data = pytesseract.image_to_data(adaptive_thresh, config=custom_config, output_type=Output.DICT)

        current_line = ""
        current_bbox = [float('inf'), float('inf'), 0, 0]  # left, top, right, bottom
        confidence = []

        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                left = data['left'][i]
                top = data['top'][i]
                width = data['width'][i]
                height = data['height'][i]
                
                # new block triggering condition
                if current_line and top > current_bbox[1] + current_bbox[3]:
                    block = {
                        "BlockType": "LINE",
                        "Text": current_line.strip(),
                        "Confidence": np.nanmean(confidence),
                        "Geometry": {
                            "BoundingBox": {
                                "Left": current_bbox[0] / img.shape[1],
                                "Top": current_bbox[1] / img.shape[0],
                                "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                                "Height": current_bbox[3] / img.shape[0]
                            }
                        },
                        "Id": block_id,
                        "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
                    }
                    blocks.append(block)
                    block_id += 1
                    # reset below counters and containers
                    current_line = ""
                    current_bbox = [float('inf'), float('inf'), 0, 0]
                    confidence = []
                
                current_line += data['text'][i] + " "
                confidence.append(data['conf'][i]) 
                current_bbox[0] = min(current_bbox[0], left)
                current_bbox[1] = min(current_bbox[1], top)
                current_bbox[2] = max(current_bbox[2], left + width)
                current_bbox[3] = max(current_bbox[3], height)
        
        # add the last line into the last new block since it is skipped in the loop above to form a new block.
        if current_line:
            block = {
                "BlockType": "LINE",
                "Text": current_line.strip(),
                "Confidence": np.nanmean(confidence),
                "Geometry": {
                    "BoundingBox": {
                        "Left": current_bbox[0] / img.shape[1],
                        "Top": current_bbox[1] / img.shape[0],
                        "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                        "Height": current_bbox[3] / img.shape[0]
                    }
                },
                "Id": block_id,
                "Page": page_num + 1 # to be consistent with block_id, both are 1-indexed.
            }
            blocks.append(block)
            block_id += 1
                
    return json.dumps(blocks, indent=2)

# Specify the path to your PDF file
pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

# Parse the PDF and get JSON output
json_output = parse_pdf(pdf_path)
print(json_output)

# COMMAND ----------

# DBTITLE 1,line level v4 (still working)
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Function to convert PDF to images and parse text using PyTesseract
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    block_id = 1
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert to BGR format for OpenCV
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # custom_config = r'--psm 6'
        # data = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        lines = {}
        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                line_num = data['line_num'][i]
                if line_num not in lines:
                    lines[line_num] = {
                        "BlockType": "LINE",
                        "text": "",
                        "left": float('inf'),
                        "top": float('inf'),
                        "right": 0,
                        "bottom": 0,
                        "confidences": [],
                        "Id": block_id
                    }
                lines[line_num]["text"] += data['text'][i] + " "
                lines[line_num]["left"] = min(lines[line_num]["left"], data['left'][i])
                lines[line_num]["top"] = min(lines[line_num]["top"], data['top'][i])
                lines[line_num]["right"] = max(lines[line_num]["right"], data['left'][i] + data['width'][i])
                lines[line_num]["bottom"] = max(lines[line_num]["bottom"], data['top'][i] + data['height'][i])
                lines[line_num]["confidences"].append(data['conf'][i])

        blocks.append(block)
        block_id += 1

        # current_line = ""
        # current_bbox = [float('inf'), float('inf'), 0, 0]  # left, top, right, bottom
        
        # for i in range(len(data['text'])):
        #     if data['text'][i].strip() != '':
        #         left = data['left'][i]
        #         top = data['top'][i]
        #         width = data['width'][i]
        #         height = data['height'][i]
                
        #         if current_line and top > current_bbox[1] + current_bbox[3]:
                    block = {
                        "BlockType": "LINE",
                        "Text": current_line.strip(),
                        "Confidence": sum(data['conf'][j] for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]) / len([j for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]]),
                        "Geometry": {
                            "BoundingBox": {
                                "Left": current_bbox[0] / img.shape[1],
                                "Top": current_bbox[1] / img.shape[0],
                                "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                                "Height": current_bbox[3] / img.shape[0]
                            }
                        },
                        "Id": block_id
                    }
                    blocks.append(block)
                    block_id += 1
                    current_line = ""
                    current_bbox = [float('inf'), float('inf'), 0, 0]
                
                current_line += data['text'][i] + " "
                current_bbox[0] = min(current_bbox[0], left)
                current_bbox[1] = min(current_bbox[1], top)
                current_bbox[2] = max(current_bbox[2], left + width)
                current_bbox[3] = max(current_bbox[3], height)
        
        if current_line:
            block = {
                "BlockType": "LINE",
                "Text": current_line.strip(),
                "Confidence": sum(data['conf'][j] for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]) / len([j for j in range(len(data['text'])) if data['top'][j] == current_bbox[1]]),
                "Geometry": {
                    "BoundingBox": {
                        "Left": current_bbox[0] / img.shape[1],
                        "Top": current_bbox[1] / img.shape[0],
                        "Width": (current_bbox[2] - current_bbox[0]) / img.shape[1],
                        "Height": current_bbox[3] / img.shape[0]
                    }
                },
                "Id": block_id
            }
            blocks.append(block)
            block_id += 1
                
    return json.dumps(blocks, indent=2)

# Specify the path to your PDF file
pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"

# Parse the PDF and get JSON output
json_output = parse_pdf(pdf_path)
print(json_output)

# COMMAND ----------

# DBTITLE 1,visualiza func backup
# import fitz  # PyMuPDF
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to display a chosen page with bounding box overlay
# def display_page_with_bboxes(pdf_path, json_output, page_number):
#     # Load the PDF
#     doc = fitz.open(pdf_path)
#     page = doc.load_page(page_number)
#     pix = page.get_pixmap()
#     img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
#     # Convert to BGR format for OpenCV
#     if pix.n == 4:
#         img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#     elif pix.n == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
#     # Load the JSON output
#     blocks = json.loads(json_output)
    
#     # Draw bounding boxes
#     for block in blocks:
#         bbox = block['Geometry']['BoundingBox']
#         left = int(bbox['Left'] * img.shape[1])
#         top = int(bbox['Top'] * img.shape[0])
#         width = int(bbox['Width'] * img.shape[1])
#         height = int(bbox['Height'] * img.shape[0])
        
#         # Draw rectangle
#         cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 1)
    
#     # Display the image with bounding boxes
#     plt.figure(figsize=(10, 10))
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

# # # Specify the path to your PDF file and JSON output
# # pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"
# # json_output = parse_pdf(pdf_path)  # Assuming parse_pdf function is defined in previous cells

# COMMAND ----------

# DBTITLE 1,visualiza func with page number
import fitz  # PyMuPDF
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display a chosen page with bounding box overlay
def display_page_with_bboxes(pdf_path, json_output, page_number):
    # Load the PDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1) # to be consistent with 1-index in json block "Page"
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    # Convert to BGR format for OpenCV
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Load the JSON output
    blocks = json.loads(json_output)
    
    # Draw bounding boxes
    for block in blocks:
        if block['Page'] == page_number:
            bbox = block['Geometry']['BoundingBox']
            left = int(bbox['Left'] * img.shape[1])
            top = int(bbox['Top'] * img.shape[0])
            width = int(bbox['Width'] * img.shape[1])
            height = int(bbox['Height'] * img.shape[0])
            
            # Draw rectangle
            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 1)

            # annotate with block id
            cv2.putText(img, f"Id: {block['Id']}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # also print validation info
            print(f"Page: {block['Page']}")
            print(f"Block type: {block['BlockType']}")
            print(f"Block ID: {block['Id']}")
            print(f"Confidence: {block['Confidence']}")
            print(f"Block text: {block['Text']}")
            print(f"Bounding box: {bbox}")
            print("-"*20)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 15))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# # Specify the path to your PDF file and JSON output
# pdf_path = f"{landing_zone}/{'W291WIE02498.pdf'}"
# json_output = parse_pdf(pdf_path)  # Assuming parse_pdf function is defined in previous cells

# COMMAND ----------

# Display the chosen page with bounding boxes
page_number = int(input("Enter the page number: "))
display_page_with_bboxes(pdf_path, json_output, page_number=page_number)

# COMMAND ----------

# MAGIC %md
# MAGIC # Supplemental

# COMMAND ----------

# %sh
# #: (DONT RUN) poppler-utils are out of date.
# sudo apt-get install -y tesseract-ocr 
# sudo apt-get install -y libpoppler-cpp-dev pkg-config poppler-utils

# COMMAND ----------

# MAGIC %sh
# MAGIC # sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get purge && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y
