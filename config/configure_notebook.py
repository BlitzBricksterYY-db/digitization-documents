# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /dbfs/FileStore/solution_accelerators/digitization/ && touch /dbfs/FileStore/solution_accelerators/digitization/init.sh
# MAGIC
# MAGIC # replace init.sh path with your UC volumn path
# MAGIC cat <<EOF > /dbfs/FileStore/solution_accelerators/digitization/init.sh
# MAGIC #!/usr/bin/env bash
# MAGIC sudo rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get purge && sudo apt-get clean && sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr -y
# MAGIC # sudo apt-get install -y tesseract-ocr
# MAGIC # sudo apt-get install -y poppler-utils # obsolete package, pip pdf2image does the job
# MAGIC EOF
# MAGIC
# MAGIC # copy to volume for UC path
# MAGIC cp -p '/dbfs/FileStore/solution_accelerators/digitization/init.sh' '/Volumes/yyang/centene_testing/init-scripts/init.sh'

# COMMAND ----------

# DBTITLE 1,added source the init.sh
# MAGIC %sh
# MAGIC # this is only needed for local interactive notebook testing, it should be setup as init.sh for your cluster
# MAGIC source /Volumes/yyang/centene_testing/init-scripts/init.sh

# COMMAND ----------

# Set mlflow experiment explicitly to make sure the code runs in both interactive execution and job execution
import mlflow
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/document_digitization'.format(username))
model_name = 'table_classification'

# COMMAND ----------

# Set sector to include brewing companies， for healthcare payor, it is i91.
sector = 'i22' 

# Here we use a `/tmp/...` path in DBFS to minimize dependency. We recommend using a `/mnt/...` path or one that directly connects to your cloud storage for production usage. To learn more about mount points, please review [this document](https://docs.databricks.com/dbfs/mounts.html).  If you would like to use a mount point or a different path, please update the variable below with the appropriate path:
# Here we update to use Unity Catalog path.
landing_zone = '/Volumes/yyang/centene_testing/pdf' #'/tmp/fsi/datasets/digitization/csr/files' 
k = 3
landing_zone_fs = '{}/**/pages'.format(landing_zone)

# COMMAND ----------


