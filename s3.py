import io
import cv2
import boto3


S3_REGION = 'ap-northeast-2'
S3_BUCKET = 'crack-detected-data'
"""
****************** 중요  ************************
1. S3_KEY & S#_SECRET은 로컬에서 프로젝트 실행 시 알맞는 값으로 변경 바람. 
2. git commit과 push할 때 반드시 none으로 설정해줄 것.
Ex) S3_KEY = "none"
*************************************************
"""
S3_KEY = "none"
S3_SECRET = "none"


s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
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
    return s3_url


# def get_s3url(url):