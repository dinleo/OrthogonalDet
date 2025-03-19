from dotenv import load_dotenv
import os
from huggingface_hub import HfApi

load_dotenv()
HUG = os.getenv('HUG')
api = HfApi()

api.upload_folder(
    folder_path='output', # 폴더 안 내용물만 들어간다.
    repo_id="dinleo11/OrthogonalDet", # 레포 주소
    path_in_repo="kaggle/train_t1", # 레포 내 저장할 폴더
    repo_type="model",
    token=HUG
)

# f = api.hf_hub_download(
#     repo_id="dinleo11/OrthogonalDet",
#     filename="t1.pth",
#     subfolder="kaggle/train_30000/M-OWODB",
#     repo_type="model",
#     token=HUG,
#     local_dir='./'
# )