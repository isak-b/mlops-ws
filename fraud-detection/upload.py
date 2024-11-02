import os
import boto3
import botocore

# Paths
storage_path = os.environ.get("STORAGE_PATH")
if storage_path is None:
    is_local = os.getenv("LOCAL", "false").lower() == "true"
    storage_path = "." if is_local else "/opt/app-root/src/mlops-ws/"
models_path = os.path.join(storage_path, "models/")

aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
region_name = os.environ.get("AWS_DEFAULT_REGION")
bucket_name = os.environ.get("AWS_S3_BUCKET")

if not all([aws_access_key_id, aws_secret_access_key, endpoint_url, region_name, bucket_name]):
    raise ValueError(
        "One or data connection variables are empty.  " "Please check your data connection to an S3 bucket."
    )

session = boto3.session.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

s3_resource = session.resource(
    "s3", config=botocore.client.Config(signature_version="s3v4"), endpoint_url=endpoint_url, region_name=region_name
)

bucket = s3_resource.Bucket(bucket_name)


def upload_directory_to_s3(local_directory, s3_prefix):
    num_files = 0
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path)
            print(f"{file_path} -> {s3_key}")
            bucket.upload_file(file_path, s3_key)
            num_files += 1
    return num_files


def list_objects(prefix):
    filter = bucket.objects.filter(Prefix=prefix)
    for obj in filter.all():
        print(obj.key)


list_objects("models")

if not os.path.isdir(models_path):
    raise ValueError(
        f"The directory '{models_path}' does not exist.  " "Did you finish training the model in the previous notebook?"
    )

num_files = upload_directory_to_s3(models_path, "models")

if num_files == 0:
    raise ValueError(
        "No files uploaded.  Did you finish training and "
        'saving the model to the "models" directory?  '
        'Check for "models/fraud/1/model.onnx"'
    )
