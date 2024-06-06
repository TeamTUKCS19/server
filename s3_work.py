import io
import cv2
import boto3
from s3_secret import S3_SECRET, S3_KEY

S3_REGION = 'ap-northeast-2'
S3_BUCKET = 'crack-detected-data'
s3_KEY = S3_KEY
s3_SECRET = S3_SECRET

s3_client = boto3.client(
    's3',
    aws_access_key_id=s3_KEY,
    aws_secret_access_key=s3_SECRET,
    region_name=S3_REGION
)


# 버킷 생성

# 버킷에 파일 업로드
def upload_to_s3(image, filename):
    # 메모리 내에서 이미지를 JPEG 형식으로 인코딩
    _, buffer = cv2.imencode('.jpg', image)
    image_byte = io.BytesIO(buffer)

    # Upload to S3
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=filename,
        Body=image_byte.getvalue(),
        ContentType='image/jpeg'
    )
    s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
    print(s3_url)
    return s3_url
