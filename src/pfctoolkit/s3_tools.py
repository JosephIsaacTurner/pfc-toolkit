import boto3
from urllib.parse import urljoin
import gzip
from io import BytesIO
import nibabel as nib
import numpy as np

class S3Storage:

    def __init__(self, config):
        self.session = boto3.session.Session()
        self.AWS_STORAGE_BUCKET_NAME = config.get("AWS_STORAGE_BUCKET_NAME")
        self.AWS_S3_ENDPOINT_URL = config.get("AWS_S3_ENDPOINT_URL")
        self.AWS_ACCESS_KEY_ID = config.get("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = config.get("AWS_SECRET_ACCESS_KEY")
    
    def save(self, name, content, max_length=None):
        headers = {'ContentType': ''}
        s3 = self._get_s3_client()
        name = name.replace("s3://","").replace(self.AWS_STORAGE_BUCKET_NAME+"/","")
        try:
            s3.upload_fileobj(content, self.AWS_STORAGE_BUCKET_NAME, name, ExtraArgs=headers)
        except Exception as e:
            print(f"Failed to upload: {e}")
            return None
        return name

    def url(self, name):
        return urljoin(self.MEDIA_URL, f"{name}")

    def get_file_from_cloud(self, cloud_filepath):
        """Load file from S3 and convert to appropriate format based on extension.

        Parameters:
        - cloud_filepath (str): Path to the file in S3

        Returns:
        - Depending on the file type:
            - nib.Nifti1Image: for .gz (assuming .nii.gz)
            - np.ndarray: for .npy
        """

        # Extract file extension
        extension = cloud_filepath.split('.')[-1]
        
        # Create an S3 client
        client = self._get_s3_client()

        # Extract bucket name and key from the cloud_filepath
        s3_components = cloud_filepath[5:].split('/', 1)
        bucket_name = s3_components[0]
        key = s3_components[1] if len(s3_components) > 1 else ""
        
        try:
            file_object = client.get_object(Bucket=bucket_name, Key=key)
            file_data = file_object['Body'].read()
        except Exception as e:
            print(f"Failed to fetch: {e}")
            return None

        if extension == 'gz':
            fh = nib.FileHolder(fileobj=gzip.GzipFile(fileobj=BytesIO(file_data)))
            return nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
        elif extension == 'npy':
            return np.load(BytesIO(file_data), allow_pickle=True)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def _get_s3_client(self):
        return self.session.client('s3',
                                   region_name='nyc3',
                                   endpoint_url=self.AWS_S3_ENDPOINT_URL,
                                   aws_access_key_id=self.AWS_ACCESS_KEY_ID,
                                   aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY)
    
    def compress_nii_image(self, nii_img):
        """Compresses an NIfTI image.

        Args:
        - nii_img: An instance that has a to_bytes method which converts the image to byte data.

        Returns:
        - A BytesIO object containing the compressed image data.
        """
        img_data = nii_img.to_bytes()
        img_data_gz = BytesIO()
        with gzip.GzipFile(fileobj=img_data_gz, mode='w') as f_out:
            f_out.write(img_data)
        img_data_gz.seek(0)
        return img_data_gz
    
    def list_s3_files(self, s3_path):
        """
        List NIfTI files in an S3 bucket directory.

        Parameters
        ----------
        s3_path : str
            Path in the format s3://bucket-name/prefix
        
        Returns
        -------
        roi_paths : list of str
            List of S3 paths to NIfTI image ROIs.
        """

        # Parse the s3_path to extract bucket name and prefix
        if not s3_path.startswith('s3://'):
            raise ValueError("Provided path is not a valid S3 path.")
        
        s3_components = s3_path[5:].split('/', 1)
        bucket_name = s3_components[0]
        prefix = s3_components[1] if len(s3_components) > 1 else ""

        # Use the S3 client from the class's method
        s3 = self._get_s3_client()
        
        roi_paths = []
        
        paginator = s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                if (obj['Key'].endswith('.nii') or obj['Key'].endswith('.nii.gz')):
                    roi_paths.append('s3://' + bucket_name + '/' + obj['Key'])
        
        return roi_paths
