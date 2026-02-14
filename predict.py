import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import List, Optional

import requests
import websocket
from cog import BasePredictor, Input, Path, Secret


WORKFLOW_PATH = "srpo_workflow_api.json"
COMFY_HOST = "127.0.0.1:8188"
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
TEMP_DIR = "ComfyUI/temp"

CHECKPOINT_FILENAME = "srpoRefineQuantizedFp16_v10Fp16.safetensors"
VAE_FILENAME = "UltraFlux-v1.safetensors"
CLIP_L_FILENAME = "clip_l.safetensors"
T5_FILENAME = "t5xxl_fp16.safetensors"

CHECKPOINT_URL = "https://huggingface.co/Maikoke/srpo-base/resolve/main/srpoRefineQuantizedFp16_v10Fp16.safetensors"
VAE_URL = "https://huggingface.co/Owen777/UltraFlux-v1/resolve/main/vae/diffusion_pytorch_model.safetensors"
CLIP_L_URL = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
T5_URL = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_query_param(url: str, key: str, value: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    query[key] = [value]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def normalize_civitai_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if "civitai.com" not in parsed.netloc:
        return url

    if parsed.path.startswith("/models"):
        query = urllib.parse.parse_qs(parsed.query)
        model_version_id = query.get("modelVersionId", [None])[0]
        if model_version_id:
            return (
                f"https://civitai.com/api/download/models/{model_version_id}"
                "?type=Model&format=SafeTensor"
            )
    return url


def redact_url_for_log(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        sensitive_keys = {
            "token",
            "api_key",
            "apikey",
            "key",
            "auth",
            "authorization",
            "access_token",
        }
        for key in list(query.keys()):
            if key.lower() in sensitive_keys:
                query[key] = ["***"]
        return urllib.parse.urlunparse(
            parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
        )
    except Exception:
        return url


def streamed_download(
    url: str,
    destination: str,
    headers: Optional[dict] = None,
    max_retries: int = 3,
    expected_min_size: int = 0,
) -> None:
    """Download a file with retries, progress logging, and content validation.

    Args:
        url: URL to download from.
        destination: Local path to save the file.
        headers: Extra HTTP headers.
        max_retries: Number of download attempts.
        expected_min_size: Minimum expected file size in bytes.  If the
            downloaded file is smaller than this, the download is considered
            failed (likely an error page or redirect to login).
    """
    ensure_dir(os.path.dirname(destination))
    tmp_destination = destination + ".part"

    default_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    if headers:
        default_headers.update(headers)

    last_error: Optional[Exception] = None
    redacted_url = redact_url_for_log(url)
    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[download] Attempt {attempt}/{max_retries}: "
                f"{redacted_url} -> {destination}"
            )
            with requests.get(
                url,
                stream=True,
                timeout=(30, 1800),
                headers=default_headers,
                allow_redirects=True,
            ) as response:
                response.raise_for_status()

                # --- Content-type validation ---
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type or "text/plain" in content_type:
                    # Read a small snippet for the error message
                    snippet = next(response.iter_content(chunk_size=512), b"")
                    raise RuntimeError(
                        f"Server returned content-type '{content_type}' instead of a "
                        f"binary file. This usually means authentication is required "
                        f"or the URL is invalid. Response preview: "
                        f"{snippet[:200]!r}"
                    )

                total = response.headers.get("content-length")
                total_bytes = int(total) if total else None
                downloaded = 0
                last_log_time = time.time()

                with open(tmp_destination, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
                            # Log progress every 10 seconds
                            now = time.time()
                            if now - last_log_time >= 10:
                                dl_mb = downloaded / (1024 * 1024)
                                if total_bytes:
                                    pct = downloaded / total_bytes * 100
                                    total_mb = total_bytes / (1024 * 1024)
                                    print(
                                        f"[download] Progress: {dl_mb:.1f} / "
                                        f"{total_mb:.1f} MB ({pct:.1f}%)"
                                    )
                                else:
                                    print(f"[download] Progress: {dl_mb:.1f} MB")
                                last_log_time = now

                if total_bytes and total_bytes > 0 and downloaded < total_bytes:
                    raise RuntimeError(
                        f"Incomplete download: got {downloaded} bytes, "
                        f"expected {total_bytes}"
                    )

                # --- Post-download size validation ---
                if expected_min_size > 0 and downloaded < expected_min_size:
                    raise RuntimeError(
                        f"Downloaded file is too small: {downloaded} bytes, "
                        f"expected at least {expected_min_size} bytes "
                        f"({expected_min_size / (1024*1024):.1f} MB). "
                        f"The URL likely returned an error page or requires "
                        f"authentication."
                    )

            os.replace(tmp_destination, destination)
            size_mb = os.path.getsize(destination) / (1024 * 1024)
            print(f"[download] Success: {destination} ({size_mb:.1f} MB)")
            return
        except Exception as exc:
            last_error = exc
            print(f"[download] Attempt {attempt} failed: {exc}")
            if os.path.exists(tmp_destination):
                os.remove(tmp_destination)
            if attempt < max_retries:
                time.sleep(5 * attempt)

    raise RuntimeError(
        f"Failed to download {redacted_url} after {max_retries} attempts: {last_error}"
    )


def ensure_downloaded(
    url: str,
    destination: str,
    headers: Optional[dict] = None,
    min_size: int = 1024,
) -> None:
    if os.path.exists(destination) and os.path.getsize(destination) >= min_size:
        print(f"[download] Already exists: {destination}")
        return
    streamed_download(
        url=url,
        destination=destination,
        headers=headers,
        expected_min_size=min_size,
    )


def list_files_recursive(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []

    files: List[str] = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            files.append(os.path.join(root, name))
    return sorted(files)


class ComfyRunner:
    def __init__(self, host: str):
        self.host = host
        self.server_process: Optional[subprocess.Popen] = None

    def start_server(self) -> None:
        command = [
            "python",
            "ComfyUI/main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
            "--output-directory",
            OUTPUT_DIR,
            "--input-directory",
            INPUT_DIR,
            "--disable-metadata",
        ]

        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if self.server_process.stdout is not None:
            threading.Thread(target=self._pipe_logs, args=(self.server_process.stdout,), daemon=True).start()

        start = time.time()
        while not self.is_server_running():
            if time.time() - start > 300:
                raise RuntimeError("ComfyUI server did not start within 300 seconds")
            time.sleep(1)

    @staticmethod
    def _pipe_logs(stream) -> None:
        for line in iter(stream.readline, ""):
            print(f"[ComfyUI] {line.rstrip()}")

    def is_server_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://{self.host}/history/123", timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def clear_queue(self) -> None:
        payload = json.dumps({"clear": True}).encode("utf-8")
        request = urllib.request.Request(
            f"http://{self.host}/queue",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=10).read()

        interrupt = urllib.request.Request(f"http://{self.host}/interrupt", data=b"", method="POST")
        urllib.request.urlopen(interrupt, timeout=10).read()

    def run_workflow(self, workflow: dict) -> None:
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.host}/ws?clientId={client_id}", timeout=30)

        try:
            payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
            request = urllib.request.Request(
                f"http://{self.host}/prompt",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = json.loads(urllib.request.urlopen(request, timeout=30).read())
            prompt_id = response["prompt_id"]

            while True:
                raw_message = ws.recv()
                if not isinstance(raw_message, str):
                    continue
                message = json.loads(raw_message)

                if message.get("type") == "execution_error":
                    error_data = message.get("data", {})
                    raise RuntimeError(f"Workflow execution failed: {json.dumps(error_data)}")

                if message.get("type") == "executing":
                    data = message.get("data", {})
                    if data.get("prompt_id") == prompt_id and data.get("node") is None:
                        break
        finally:
            ws.close()


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("[setup] Starting setup...")
        ensure_dir(OUTPUT_DIR)
        ensure_dir(INPUT_DIR)
        ensure_dir(TEMP_DIR)

        ensure_dir("ComfyUI/models/checkpoints")
        ensure_dir("ComfyUI/models/vae")
        ensure_dir("ComfyUI/models/clip")
        ensure_dir("ComfyUI/models/loras")

        with open(WORKFLOW_PATH, "r", encoding="utf-8") as file:
            self.base_workflow = json.load(file)

        self.comfy = ComfyRunner(COMFY_HOST)
        self._ready_lock = threading.Lock()
        self._is_ready = False
        print("[setup] Setup complete (lazy runtime init enabled).")

    def _ensure_base_models_downloaded(self, civitai_token: str = "") -> None:
        print("[runtime-init] Ensuring base models are downloaded...")

        checkpoint_url = CHECKPOINT_URL
        checkpoint_host = urllib.parse.urlparse(CHECKPOINT_URL).netloc.lower()
        if "civitai.com" in checkpoint_host:
            # CivitAI requires an API token for most model downloads.
            # Priority: prediction input token -> environment variable.
            civitai_token = civitai_token.strip() or os.environ.get("CIVITAI_API_TOKEN", "").strip()
            if civitai_token:
                checkpoint_url = append_query_param(CHECKPOINT_URL, "token", civitai_token)
                print("[runtime-init] Using CivitAI API token for checkpoint download.")
            else:
                print(
                    "[runtime-init] WARNING: No CivitAI token provided in input and "
                    "no CIVITAI_API_TOKEN env var set. "
                    "CivitAI downloads may fail if authentication is required."
                )

        ensure_downloaded(
            checkpoint_url,
            os.path.join("ComfyUI/models/checkpoints", CHECKPOINT_FILENAME),
            min_size=1024 * 1024 * 100,  # SRPO checkpoint should be >100 MB
        )
        ensure_downloaded(
            VAE_URL,
            os.path.join("ComfyUI/models/vae", VAE_FILENAME),
            min_size=1024 * 1024 * 100,  # VAE should be >100 MB
        )
        ensure_downloaded(
            CLIP_L_URL,
            os.path.join("ComfyUI/models/clip", CLIP_L_FILENAME),
            min_size=1024 * 1024 * 100,  # CLIP_L should be >100 MB
        )
        ensure_downloaded(
            T5_URL,
            os.path.join("ComfyUI/models/clip", T5_FILENAME),
            min_size=1024 * 1024 * 1000,  # T5 should be >1 GB
        )
        print("[runtime-init] Base models ready.")

    def _ensure_ready(self, civitai_token: str = "") -> None:
        if self._is_ready and self.comfy.is_server_running():
            return

        with self._ready_lock:
            if self._is_ready and self.comfy.is_server_running():
                return

            print("[runtime-init] Initializing runtime dependencies...")
            self._ensure_base_models_downloaded(civitai_token=civitai_token)

            if not self.comfy.is_server_running():
                print("[runtime-init] Starting ComfyUI server...")
                self.comfy.start_server()

            self._is_ready = True
            print("[runtime-init] Runtime initialization complete.")

    @staticmethod
    def _cleanup_prediction_dirs() -> None:
        for directory in [OUTPUT_DIR, INPUT_DIR, TEMP_DIR]:
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            ensure_dir(directory)

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _download_lora(
        self,
        lora_url: str,
        civitai_api_key: Optional[Secret] = None,
        huggingface_token: Optional[Secret] = None,
    ) -> str:
        url = normalize_civitai_url(lora_url.strip())

        parsed = urllib.parse.urlparse(url)
        if "civitai.com" in parsed.netloc:
            # Try per-request key first, then env var fallback
            key = ""
            if civitai_api_key is not None:
                key = civitai_api_key.get_secret_value().strip()
            if not key:
                key = os.environ.get("CIVITAI_API_TOKEN", "").strip()
            if key:
                url = append_query_param(url, "token", key)

        headers = {}
        if "huggingface.co" in parsed.netloc and huggingface_token is not None:
            token = huggingface_token.get_secret_value().strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        extension = os.path.splitext(parsed.path)[1].lower()
        if extension not in [".safetensors", ".pt", ".pth"]:
            extension = ".safetensors"

        filename = f"lora_{self._hash_url(url)}{extension}"
        destination = os.path.join("ComfyUI/models/loras", filename)
        ensure_downloaded(
            url=url,
            destination=destination,
            headers=headers,
            min_size=1024 * 100,  # LoRA should be at least ~100 KB
        )
        return filename

    def _build_workflow(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        guidance: float,
        seed: int,
        lora_name: str,
        lora_strength: float,
    ) -> dict:
        workflow = copy.deepcopy(self.base_workflow)

        # Ensure output node is SaveImage (not PreviewImage)
        workflow["559"]["class_type"] = "SaveImage"
        workflow["559"]["inputs"]["filename_prefix"] = "srpo"

        workflow["567"]["inputs"]["ckpt_name"] = CHECKPOINT_FILENAME
        workflow["570"]["inputs"]["vae_name"] = VAE_FILENAME

        workflow["455"]["inputs"]["text"] = prompt
        workflow["428"]["inputs"]["text"] = negative_prompt

        workflow["566"]["inputs"]["width"] = width
        workflow["566"]["inputs"]["height"] = height
        workflow["118"]["inputs"]["width"] = width
        workflow["118"]["inputs"]["height"] = height

        workflow["119"]["inputs"]["guidance"] = guidance
        workflow["110"]["inputs"]["seed"] = seed
        workflow["110"]["inputs"]["steps"] = steps
        workflow["110"]["inputs"]["cfg"] = cfg

        workflow["576"]["inputs"]["lora_name"] = lora_name
        workflow["576"]["inputs"]["strength_model"] = lora_strength

        return workflow

    def predict(
        self,
        prompt: str = Input(description="Positive prompt", default="professional headshot"),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="bad image, bad photo, bad hand, bad finger, logo, Backlight, worst quality, low resolution, distorted, twisted, watermark.",
        ),
        lora_url: str = Input(description="Direct download URL to LoRA (.safetensors preferred)"),
        lora_strength: float = Input(description="LoRA strength on model", default=1.0, ge=0.0, le=2.0),
        width: int = Input(description="Image width", default=1024, ge=512, le=1536),
        height: int = Input(description="Image height", default=1024, ge=512, le=1536),
        steps: int = Input(description="Sampling steps", default=25, ge=1, le=100),
        cfg: float = Input(description="CFG scale", default=1.0, ge=0.1, le=20.0),
        guidance: float = Input(description="Flux guidance", default=3.5, ge=0.1, le=20.0),
        seed: Optional[int] = Input(description="Seed (random if omitted)", default=None),
        civitai_api_key: Optional[Secret] = Input(
            description=(
                "Optional CivitAI API key for private/gated LoRA URLs and "
                "for base checkpoint download auth if required"
            ),
            default=None,
        ),
        huggingface_token: Optional[Secret] = Input(
            description="Optional Hugging Face token for private/gated LoRA URLs",
            default=None,
        ),
    ) -> List[Path]:
        runtime_civitai_token = ""
        if civitai_api_key is not None:
            runtime_civitai_token = civitai_api_key.get_secret_value().strip()

        self._ensure_ready(civitai_token=runtime_civitai_token)
        self.comfy.clear_queue()
        self._cleanup_prediction_dirs()

        final_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        lora_filename = self._download_lora(
            lora_url=lora_url,
            civitai_api_key=civitai_api_key,
            huggingface_token=huggingface_token,
        )

        workflow = self._build_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            guidance=guidance,
            seed=final_seed,
            lora_name=lora_filename,
            lora_strength=lora_strength,
        )

        self.comfy.run_workflow(workflow)

        output_files = [p for p in list_files_recursive(OUTPUT_DIR) if os.path.isfile(p)]
        if not output_files:
            output_files = [p for p in list_files_recursive(TEMP_DIR) if os.path.isfile(p)]
        if not output_files:
            raise RuntimeError("No output files were produced by ComfyUI")

        return [Path(path) for path in output_files]
