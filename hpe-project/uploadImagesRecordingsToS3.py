# python3.9

# Eric Wang 06/09/2022


import boto3
import os
import io
# import re
# import pandas as pd
# import sys
# import gzip
import tarfile
from io import BytesIO
from time import time

# setup s3 and get utils
session = boto3.Session(profile_name = 'analytics-test')
client_glue = session.client('glue')
client_s3 = session.client('s3')
# sys.path.append("./library")
# from GlueJobCommonUtil import get_lake_bucket_name, get_date

# setup utils
# To retrieve the right bucket name of data lake per AWS env (-test)
def get_lake_bucket_name(client_s3):
    buckets_json = client_s3.list_buckets()
    buckets = buckets_json['Buckets']
    for bucket in buckets:
        bucket_name = bucket['Name']
        name_match = re.search('tto-analytics-.*datalake', bucket_name)
        if name_match:
            glue_bucket = name_match.group(0)
    return glue_bucket

# Get date hour using time zone and gap
def get_date_hmsz(time_zone, time_gap):
    current_time = datetime.now(timezone(time_zone))
    last_hour_from = current_time + timedelta(hours=time_gap*-1)
    last_hour_from_str = last_hour_from.strftime('%Y-%m-%d %H:%M:%S%z')
    return last_hour_from_str[0:22]+':'+last_hour_from_str[-2:]

# set paths
IMG_PATH = 'upload_test_images'
REC_PATH = 'upload_test_recordings'

# Get bucket name
glue_bucket = get_lake_bucket_name(client_s3)

# Create an daily range
last_date_from_str = get_date_hmsz('Australia/Adelaide', 0)
filename_output = last_date_from_str.replace(' ','-').replace(':','-').replace('+','_')


# Upload images
# put images as tar.gz object to s3
byte_buffer = BytesIO()

time0 = time()
# create a tar.gz file and write images as object (ByteIO object)
with tarfile.open(mode='w:gz', fileobj=byte_buffer) as targz_file:
    for filename in os.listdir(IMG_PATH):
        if '.DS_Store' not in filename:
            file = os.path.join(IMG_PATH, filename)
        targz_file.add(file)

# put the tar.gz object to S3
client_s3.put_object(Body=byte_buffer.getvalue()
                     , Bucket=glue_bucket
                     , Key='ml/eric-test/Address/images-{}.tar.gz'.format(filename_output))
print("INFO: Images successfully saved.")
print("Images upload time duration: {:.2f} minutes".format((time()-time0)/60))



# Upload recordings
# option 2
# put recordings as tar.gz object to s3
byte_buffer = BytesIO()

# create tar.gz file and write all recording files as object (BytesIO object)
time0 = time()
with tarfile.open(mode='w:gz', fileobj=byte_buffer) as targz_file:
    for filename in os.listdir(REC_PATH):
        if '.DS_Store' not in filename:
            file = os.path.join(REC_PATH, filename)
        targz_file.add(file)

# put the tar.gz object to S3
client_s3.put_object(Body=byte_buffer.getvalue()
                     , Bucket=glue_bucket
                     , Key='ml/eric-test/Address/recordings-{}.tar.gz'.format(filename_output))
print("INFO: Recordings successfully saved.")
print("Recordings upload time duration: {:.2f} minutes".format((time()-time0)/60))
