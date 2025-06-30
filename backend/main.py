import glob
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import time
import uuid

import boto3
import cv2
import ffmpegcv
import modal
import numpy as np
import whisperx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai
from pydantic import BaseModel
from tqdm import tqdm


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

def create_vertical_video(
        tracks: list, scores: list, pyframes_path: str, pyavi_path: str, 
        audio_path: str, output_path: str, framerate: int = 25
):
    target_width = 1080
    target_height = 1920

    # Find jpg files and sort them by number of frames
    frame_list = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    frame_list.sort()

    faces = [[] for _ in range(len(frame_list))]

    # Average speaker score over a window of 30 frames for smoothness
    for track_index, track in enumerate(tracks):
        score_array = scores[track_index]
        for frame_index, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(frame_index - 30, 0)
            slice_end = min(frame_index + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice) if len(score_slice) > 0 else 0)

            # Position (x, y), size (s)
            faces[frame].append({
                "track": track_index, 
                "score": avg_score, 
                "s": track["proc_track"]["s"][frame_index], 
                "x": track["proc_track"]["x"][frame_index], 
                "y": track["proc_track"]["y"][frame_index]
            })

    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")
    video_output = None
    for frame_index, frame_name in tqdm(
        enumerate(frame_list), total=len(frame_list), desc="Creating vertical video"
    ):
        image = cv2.imread(frame_name)
        if image is None:
            continue

        current_faces = faces[frame_index]
        max_score_face = max(current_faces, key=lambda face: face["score"]) if current_faces else None

        if max_score_face and max_score_face["score"] < 0:
            max_score_face = None

        if video_output is None:
            video_output = ffmpegcv.VideoWriterNV(
                file=temp_video_path, 
                codec=None, 
                fps=framerate,
                resize=(target_width, target_height)
            )

        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            scale = target_width / image.shape[1]
            resized_height = int(image.shape[0] * scale)
            resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Resize the width and height for blurred video in the background
            scale_for_background = max(target_width / image.shape[1], target_height / image.shape[0])
            background_width = int(image.shape[1] * scale_for_background)
            background_height = int(image.shape[0] * scale_for_background)
            blurred_background = cv2.resize(image, (background_width, background_height))
            blurred_background = cv2.GaussianBlur(blurred_background, (121, 121), 0)

            crop_x = (background_width - target_width) // 2
            crop_y = (background_height - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y + target_height, crop_x:crop_x + background_width]

            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y + resized_height, :] = resized_image

            video_output.write(blurred_background)
        elif mode == "crop":
            scale = target_height / image.shape[0]
            resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            # Align x coordinate with center of frame
            center_x = int(max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(min(center_x - target_width // 2, frame_width - target_width), 0)

            image_cropped = resized_image[:target_height, top_x:top_x + target_width]
            video_output.write(image_cropped)

        if video_output:
            video_output.release()

        ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path}" 
                          f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k" 
                          f"{output_path}")
        subprocess.run(ffmpeg_command, shell=True, check=True, text=True)

def process_clip(
        base_dir: str, video_path: str, s3_key: str, start_time: float, 
        end_time: float, clip_index: int, transcript_segments: list
):
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Segment clip from start to end
    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    # Cut clip
    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {video_path} -ss {start_time} -t {duration} {clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True, capture_output=True, text=True)

    # Extract audio
    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")
    columbia_cmd = (f"python Columbia_test.py --videoName {clip_name} "
                    f"--videoFolder {str(base_dir)} --pretrainModel weight/finetuning_TalkSet.model")
    
    columbia_start = time.time()
    subprocess.run(columbia_cmd, cwd="/LR-ASD", shell=True)
    columbia_duration = time.time() - columbia_start
    print(f"Columbia script duration is {columbia_duration:.2f} seconds")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"

    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")
    
    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(tracks_path, "rb") as f:
        scores = pickle.load(f)

    video_creation_start = time.time()
    create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path)
    video_creation_duration = time.time() - video_creation_start
    print(f"Clip {clip_index} vertical video creation took {video_creation_duration:.2f} seconds")

    s3_client = boto3.client("s3")
    s3_client.upload_file(vertical_mp4_path, "enzo-podcast-clipper", output_s3_key)


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

        cleaned_json_str = identified_moments_raw.strip()
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[len("```json"):].strip()
        if cleaned_json_str.endswith("```"):
            cleaned_json_str = cleaned_json_str[:-len("```")].strip()

        clip_moments = json.loads(cleaned_json_str)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error: Identified moments is not a list")
            clip_moments = []

        print(clip_moments)

        # Process the first three clips
        for index, moment in enumerate(clip_moments[:1]):
            if "start" in moment and "end" in moment:
                print(f"Processing clip {str(index)} from {str(moment["start"])} to {str(moment["end"])}")
                process_clip(
                    base_dir, video_path, s3_key, moment["start"], moment["end"], index, transcript_segments
                )

        if base_dir.exists():
            print(f"Cleaning up temp directory after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


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