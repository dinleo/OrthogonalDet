from dotenv import load_dotenv
import os
from huggingface_hub import HfApi

load_dotenv()
HUG = os.getenv('HUG')
api = HfApi()

api.upload_folder(
    folder_path='output', # 폴더 안 내용물만 들어간다.
    repo_id="dinleo11/OrthogonalDet", # 레포 주소
    path_in_repo="kaggle/1", # 레포 내 저장할 폴더
    repo_type="model",
    token=HUG
)