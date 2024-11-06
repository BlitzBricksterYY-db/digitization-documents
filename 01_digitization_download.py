# Databricks notebook source
# MAGIC %md
# MAGIC # Setup
# MAGIC For the purpose of this exercise, we will be loading some publicly available dataset containing text, images and tables. Available as PDF documents online, corporate responsiblity reports (CSR) are perfect examples of unstructured documents containing valuable insights. Compliance officers and market analysts would manually review these ESG disclosures, copy / paste relevant tables onto spreadsheets and gather as much information as possible from the text included across all different pages of different formats. This process could be automated using Apache Tika, Tesseract OCR (and additionally [AWS Textract](https://aws.amazon.com/textract/) or [John Snow Labs](https://nlp.johnsnowlabs.com/2022/09/06/finclf_augmented_esg_en.html) libraries). We will be loading all required libraries in the companion notebook (make sure to provision both scala and native libraries on your databricks cluster)

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC While our story officially starts with documents of any type stored on cloud storage, we will be programmatically accessing some CSR reports online for a given industry (configured by default to scan for Brewing companies). Please refer to https://www.responsibilityreports.com terms and conditions, enable internet connectivity from your databricks environment or replace this section with your own data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Helper Functions

# COMMAND ----------

from bs4 import BeautifulSoup
import requests


def get_organizations(sector):
    """
    Returns all organizations listed in responsibility report website
    :param int sector: the industry to search CSR reports
    """
    index_url = "https://www.responsibilityreports.com/Companies?ind={}".format(sector)
    response = requests.get(index_url)
    soup = BeautifulSoup(response.text, features="html.parser")
    csr_entries = [link.get('href') for link in soup.findAll('a')]
    organizations = [ele.split("/")[-1] for ele in csr_entries if ele.startswith('/Company/')]
    return organizations

  
def get_organization_details(organization):
    """
    Use beautiful soup to parse company page on responsibilityreports.com
    We parse the organization page to retrieve URL of last CSR report
    :param string organization: the name of the company to retrieve CSR report from
    """
    company_url = "https://www.responsibilityreports.com/Company/" + organization
    response = requests.get(company_url)
    soup = BeautifulSoup(response.text, features="html.parser")
    csr_url = ""
    # page contains the link to their most recent disclosures
    for link in soup.findAll('a'):
        data = link.get('href')
        if data.split('.')[-1] == 'pdf':
            csr_url = 'https://www.responsibilityreports.com' + data
            break
    return csr_url

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

# print("sector default from .yml config is: ",sector)
# sector = "i91"
# print("Now change to 'Healthcare' sector as:", sector)

# COMMAND ----------

# import uuid
# import os

# # reinitiate the landing zone for the download
# dbutils.fs.rm(landing_zone, True)
# dbutils.fs.mkdirs(landing_zone)
# print("Storage location for landing zone is: {}".format(landing_zone))

# csr_data = []
# organizations = get_organizations(sector)
# n = len(organizations)
# print('*'*50)
# print('Downloading reports for {} organization(s)'.format(n))
# print('*'*50)

# for i, organization in enumerate(organizations):
  
#     # retrieve CSR report for a given organization
#     try:
#         url = get_organization_details(organization)
#     except AttributeError:
#         print('Failed to retrieve report for [{}]'.format(organization))
#     if url:
#         try:
#             # generate a unique identifier and a unique path where files will be stored
#             doc_id = uuid.uuid4().hex
#             dir = '/dbfs{}/{}/pages'.format(landing_zone, doc_id)
#             os.makedirs(dir, exist_ok=True)

#             # download PDF content
#             response = requests.get(url)
#             content = response.content

#             # split PDF into individual pages
#             pages = split_pages(content)

#             # write each page individually to storage
#             for j, page_content in enumerate(pages):
#                 with open('{}/{}.pdf'.format(dir, j + 1), 'wb') as f:
#                     f.write(page_content)

#             print('[{}/{}] Downloaded report for [{}]'.format(i + 1, n, organization))
#         except:
#             print('[{}/{}] Failed to download report for [{}]'.format(i + 1, n, organization))
#             pass

# COMMAND ----------

# print("Storage location for landing zone is: {}".format(landing_zone))

# COMMAND ----------

# MAGIC %md
# MAGIC # Working on UC volume files

# COMMAND ----------

# DBTITLE 1,update landing zone to be UC volume
import os

landing_zone = "/Volumes/yyang/centene_testing/pdf"
print("Storage location for landing zone is: {}".format(landing_zone))
print("Content: {}".format(dbutils.fs.ls(landing_zone)))
print("Content: {}".format(os.listdir(landing_zone,  )))

# COMMAND ----------

from pyspark.sql.functions import input_file_name

# Read PDF files as binary
binary_df = spark.read.format('tika').load(landing_zone)
binary_df = binary_df.withColumn("path", input_file_name())

# COMMAND ----------

binary_df.display()

# COMMAND ----------


        try:
            # generate a unique identifier and a unique path where files will be stored
            doc_id = uuid.uuid4().hex
            dir = '{}/{}/pages'.format(landing_zone, doc_id)
            os.makedirs(dir, exist_ok=True)

            # open PDF content
from pypdf import PdfFileReader
import os

# Function to read PDF content
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfFileReader(file)
        content = ""
        for page_num in range(reader.getNumPages()):
            page = reader.getPage(page_num)
            content += page.extract_text()
    return content

# List all PDF files in the landing zone
pdf_files = [f for f in os.listdir(landing_zone) if f.endswith('.pdf')]

# Read content from each PDF file
pdf_contents = {}
for pdf_file in pdf_files:
    file_path = os.path.join(landing_zone, pdf_file)
    pdf_contents[pdf_file] = read_pdf(file_path)

# Display the content of the PDFs
for pdf_file, content in pdf_contents.items():
    print(f"Content of {pdf_file}:")
    print(content)
            

            # split PDF into individual pages
            pages = split_pages(content)

            # write each page individually to storage
            for j, page_content in enumerate(pages):
                with open('{}/{}.pdf'.format(dir, j + 1), 'wb') as f:
                    f.write(page_content)

            print('[{}/{}] Downloaded report and split into pages for [{}]'.format(i + 1, n, organization))
        except:
            print('[{}/{}] Failed to download report for [{}]'.format(i + 1, n, organization))
            pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Binary format
# MAGIC Spark comes with a native support for binary file. This operation returns a dataframe with content exposed as a byte array together with metadata such as file path, modification time or file size. In this notebook's companion library (see [github](https://github.com/databrickslabs/tika-ocr)), we used that format as a baseline to our project and extended its capability to include the entire suite of Tika parsers as well as Tesseract support, as reported in our next notebook.

# COMMAND ----------

print(landing_zone_fs)

# COMMAND ----------

binary_df = spark.read.format('binaryFile').load(landing_zone_fs)
display(binary_df)

# COMMAND ----------


