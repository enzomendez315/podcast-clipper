import os
import pathlib
import uuid

import boto3
import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


class ProcessVideoRequest(BaseModel):
    s3_key: str


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom", 
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf", 
                   "fc-cache -f -v"])
    .add_local_dir("LR-ASD", "/lr-asd", copy=True))

app = modal.App("podcast-clipper", image=image)

volume = modal.Volume.from_name("podcast-clipper-modal-cache", create_if_missing=True)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()


@app.cls(
    gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[
        modal.Secret.from_name("podcast-clipper-secret"), modal.Secret.from_name("aws-secret")
    ], volumes={mount_path: volume}
)
class PodcastClipper:
    @modal.enter()
    def load_model(self):
        print("Loading models")
        pass

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        #print(f"Processing video {request.s3_key}")
        s3_key = request.s3_key

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Incorrect bearer token", 
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create temporary folder
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download video file
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("enzo-podcast-clipper", s3_key, str(video_path))

        print(os.listdir(base_dir))


@app.local_entrypoint()
def main():
    import requests

    podcast_clipper = PodcastClipper()
    url = podcast_clipper.process_video.get_web_url()
    payload = {
        "s3_key": "test1/50centshort.mp4"
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)