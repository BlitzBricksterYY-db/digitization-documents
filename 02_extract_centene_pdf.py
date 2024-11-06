# Databricks notebook source
# MAGIC %md
# MAGIC # README
# MAGIC
# MAGIC Test PDF extraction solutions for Centene

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

import os

landing_zone = "/Volumes/yyang/centene_testing/pdf"
print("Storage location for landing zone is: {}".format(landing_zone))
print("Content: {}".format(dbutils.fs.ls(landing_zone)))
print("Content: {}".format(os.listdir(landing_zone,  )))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Separate pages helper functions
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


