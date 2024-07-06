import boto3

class aws_toolkit:
    def __init__(self, region_name:str='', aws_access_key_id:str='', aws_secret_access_key:str=''):
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_client = boto3.client('s3', region_name=region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    def create_bucket_s3(self, bucket_name:str=''):
        """
        NOTE: requires root level access to create a bucket.
        """
        self.s3_client.create_bucket(Bucket=bucket_name)
    
    def upload_file_s3(self, bucket_name:str='', file_path:str='', folder:str='Data'):
        file_name = file_path.split('/')[-1]
        self.s3_client.upload_file(file_path, bucket_name, Key=fr'{folder}/{file_name}')
    
    def download_file_s3(self, bucket_name:str='', object_name:str='', file_path:str=''):
        self.s3_client.download_file(bucket_name, object_name, file_path)
    
    def init_training(self, traing_args:dict={}):
        pass

        
    



access_key = 'AKIA4HKHGA4QWELUUSOA'
secret_key = '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'
aws_bot = aws_toolkit(region_name='us-west-2', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
aws_bot.download_file_s3(bucket_name='simplifine-customer1', file_path='/Users/alikavoosi/Desktop/DEMO/newpdf.pdf', folder='data')