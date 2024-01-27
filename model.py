import json
import os
import boto3
from dotenv import load_dotenv
from utils import query_endpoint
import streamlit as st
# from rembg import remove
from PIL import Image
from io import BytesIO
import base64

load_dotenv()


REGION_NAME = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = REGION_NAME
ROLE_NAME =  'roleforCDKsagemaker'

auth_arguments = {
    'aws_access_key_id':os.environ["aws_access_key_id"],
    'aws_secret_access_key':os.environ["aws_secret_access_key"],
    'region_name':REGION_NAME
}

runtime= boto3.client('sagemaker-runtime',**auth_arguments)

od_type2_endpoint_name = 'sagemaker-soln-dfd-js--Type2-2024-01-26-19-26-47-625'
od_type2_hpo_endpoint_name = 'sagemaker-soln-dfd-js--Type2-HPO-mxnet--2024-01-26-22-14-25-315'
od_type1_endpoint_name = 'sagemaker-soln-dfd-js--Type1-2024-01-26-16-54-19-431'
od_type1_hpo_endpoint_name = 'sagemaker-soln-dfd-js--Type1-HPO-2024-01-26-17-56-44-861'

categories = {1: 'crazing',
                2: 'inclusion',
                3: 'pitted_surface',
                4: 'patches',
                5: 'rolled-in_scale',
                6: 'scratches'}

image = "crazing_10.jpg"
 

st.set_page_config(layout="wide", page_title="Non compliance detection")

st.write("## Visual Inspection Automation with Pre-trained Amazon SageMaker Models")
st.write(
    ":dog: This solution detects product defects with an end-to-end Deep Learning workflow for quality control in manufacturing process. The solution takes input of product images and identifies defect regions with bounding boxes. In particular, this solution uses a pre-trained Sagemaker object detection model and fine-tune on the target dataset.:grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.savefig(buf, format="PNG", transparent = True)
    byte_im = buf.getvalue()
    
    return byte_im


def defect_detect(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)
    fixed = query_endpoint(runtime,upload,option,categories)  
    
    col2.write("Fixed Image :wrench:")
    buf = BytesIO()
    fixed.savefig(buf, format="png",transparent = True)
    
    col2.image(buf)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


option = st.sidebar.selectbox(
    'Please choose one Pre-trained Sagemaker Models : ',
    (od_type2_endpoint_name,
    od_type2_hpo_endpoint_name,
    od_type1_endpoint_name,
    od_type1_hpo_endpoint_name))

st.sidebar.write('Current model :', option)


if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        defect_detect(upload=my_upload)
# else:
    # defect_detect("./crazing_10.jpg")
