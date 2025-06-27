import json
import os
import pathlib
import subprocess
import time
import uuid

import boto3
import modal
import whisperx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai
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
        modal.Secret.from_name("podcast-clipper-secret"), 
        modal.Secret.from_name("aws-secret"), 
        modal.Secret.from_name("gemini-secret")
    ], volumes={mount_path: volume}
)
class PodcastClipper:
    @modal.enter()
    def load_model(self):
        print("Loading models...")

        self.whisperx_model = whisperx.load_model("large-v2", device="cuda", compute_type="float16")
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code="en", device="cuda")

        print("Transcription models loaded...")

        print("Creating Gemini client...")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Created Gemini client...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

        print("Starting transciption with WhisperX...")

        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        transcipt = self.whisperx_model.transcribe(audio, batch_size=16)
        result = whisperx.align(
            transcipt["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time

        print(f"Transcription and alignment took {str(duration)} seconds")
        
        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"]
                })

        return json.dumps(segments)
    
    def identify_moments(self, transcript: dict):
        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17", 
            contents="""
            This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

            Your task is to find and extract stories, or question and their corresponding answers from the transcript.
            Each clip should begin with the question and conclude with the answer.
            It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

            Please adhere to the following rules:
            - Ensure that clips do not overlap with one another.
            - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
            - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
            - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
            - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

            Avoid including:
            - Moments of greeting, thanking, or saying goodbye.
            - Non-question and answer interactions.

            If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

            The transcript is as follows:\n\n
            """ + str(transcript)
        )
        print(f"Identified moments response: {response.text}")
        return response.text

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
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

        # Get transcription
        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)

        # Identify moments for clips
        print("Identifying clip moments...")
        identified_moments_raw = self.identify_moments(transcript_segments)


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