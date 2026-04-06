import atexit
import argparse
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Iterable

import requests


ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
MODELS_DIR = ROOT / "models"
VENDOR_DIR = ROOT / "vendor"
LLAMA_VENDOR_DIR = VENDOR_DIR / "llama.cpp"
CURRENT_RELEASE_FILE = LLAMA_VENDOR_DIR / "CURRENT_RELEASE"

LLAMA_SERVER_HOST = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5001"))
MODEL_REPO = os.getenv("MODEL_REPO", "unsloth/Qwen3.5-4B-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Qwen3.5-4B-Q4_K_M.gguf")
MODEL_URL = os.getenv(
    "MODEL_URL",
    f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}?download=true",
)
LLAMA_RELEASE_API = os.getenv(
    "LLAMA_RELEASE_API",
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
)
LLAMA_SERVER_PATH = os.getenv("LLAMA_SERVER_PATH", "")
LLAMA_SERVER_EXTRA_ARGS = os.getenv("LLAMA_SERVER_EXTRA_ARGS", "")
LLAMA_N_GPU_LAYERS = os.getenv("LLAMA_N_GPU_LAYERS", "99")
HTTP_TIMEOUT = int(os.getenv("BACKEND_HTTP_TIMEOUT", "60"))
STARTUP_TIMEOUT = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "120"))
HOST_SYSTEM = platform.system().lower()
HOST_MACHINE = platform.machine().lower()


def ensure_dirs() -> None:
    for path in (DOWNLOADS_DIR, MODELS_DIR, VENDOR_DIR, LLAMA_VENDOR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, headers: dict[str, str] | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT, headers=headers) as response:
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)
    return destination


def asset_matches_platform(name: str) -> bool:
    lowered = name.lower()

    if HOST_SYSTEM == "darwin":
        if "macos" not in lowered and "apple" not in lowered and "darwin" not in lowered:
            return False
        if HOST_MACHINE in {"arm64", "aarch64"}:
            return "arm64" in lowered or "apple-silicon" in lowered
        if HOST_MACHINE in {"x86_64", "amd64"}:
            return "x64" in lowered or "x86_64" in lowered or "amd64" in lowered
        return False

    if HOST_SYSTEM == "linux":
        if "linux" not in lowered:
            return False
        if HOST_MACHINE in {"arm64", "aarch64"}:
            return "arm64" in lowered or "aarch64" in lowered
        if HOST_MACHINE in {"x86_64", "amd64"}:
            return "x64" in lowered or "x86_64" in lowered or "amd64" in lowered
        return False

    return False


def is_matching_asset(name: str) -> bool:
    lowered = name.lower()
    if not asset_matches_platform(name):
        return False
    if not (lowered.endswith(".zip") or lowered.endswith(".tar.gz") or lowered.endswith(".tgz")):
        return False
    return "server" in lowered or "bin" in lowered or "metal" in lowered


def select_release_asset(assets: Iterable[dict]) -> dict:
    candidates = [asset for asset in assets if is_matching_asset(asset.get("name", ""))]
    if not candidates:
        raise RuntimeError(
            f"Could not find a llama.cpp release asset for system={HOST_SYSTEM} arch={HOST_MACHINE}."
        )

    def score(asset: dict) -> tuple[int, int]:
        name = asset.get("name", "").lower()
        return (
            int("metal" in name) + int("server" in name) + int("bin" in name),
            len(name),
        )

    return sorted(candidates, key=score, reverse=True)[0]


def extract_archive(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
    elif archive_path.name.endswith(".tar.gz") or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path.name}")

    return destination


def find_llama_server(search_root: Path) -> Path:
    for candidate in search_root.rglob("llama-server"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("llama-server was not found in the extracted llama.cpp bundle.")


def load_cached_release_tag() -> str | None:
    if not CURRENT_RELEASE_FILE.exists():
        return None
    tag = CURRENT_RELEASE_FILE.read_text().strip()
    return tag or None


def store_cached_release_tag(tag_name: str) -> None:
    CURRENT_RELEASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_RELEASE_FILE.write_text(tag_name + "\n")


def fetch_latest_release() -> dict:
    response = requests.get(LLAMA_RELEASE_API, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


def ensure_llama_server(update: bool = False) -> Path:
    if LLAMA_SERVER_PATH:
        path = Path(LLAMA_SERVER_PATH).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Configured LLAMA_SERVER_PATH does not exist: {path}")
        return path

    cached_tag = load_cached_release_tag()
    if cached_tag and not update:
        extract_dir = LLAMA_VENDOR_DIR / cached_tag
        if extract_dir.exists():
            server_path = find_llama_server(extract_dir)
            print(
                f"Using cached llama.cpp release {cached_tag} at {server_path} "
                f"for system={HOST_SYSTEM} arch={HOST_MACHINE}"
            )
            return server_path

    release = fetch_latest_release()
    asset = select_release_asset(release.get("assets", []))
    archive_name = asset["name"]
    archive_path = DOWNLOADS_DIR / archive_name
    extract_dir = LLAMA_VENDOR_DIR / release["tag_name"]

    if not archive_path.exists():
        print(f"Downloading llama.cpp release asset: {archive_name}")
        download_file(asset["browser_download_url"], archive_path)

    if not extract_dir.exists():
        print(f"Extracting llama.cpp into {extract_dir}")
        extract_archive(archive_path, extract_dir)

    server_path = find_llama_server(extract_dir)
    store_cached_release_tag(release["tag_name"])
    print(f"Using llama-server at {server_path} for system={HOST_SYSTEM} arch={HOST_MACHINE}")
    return server_path


def ensure_model() -> Path:
    model_path = MODELS_DIR / MODEL_FILE
    if model_path.exists():
        print(f"Using existing model at {model_path}")
        return model_path

    print(f"Downloading model {MODEL_FILE} from {MODEL_REPO}")
    headers = {}
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    download_file(MODEL_URL, model_path, headers=headers or None)
    return model_path


def wait_for_llama(base_url: str) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    probe_urls = (f"{base_url}/health", f"{base_url}/v1/models")
    while time.time() < deadline:
        for url in probe_urls:
            try:
                response = requests.get(url, timeout=2)
                if response.ok:
                    return
            except requests.RequestException:
                pass
        time.sleep(1)
    raise RuntimeError(f"llama-server did not become ready within {STARTUP_TIMEOUT} seconds.")


def start_llama_server(server_path: Path, model_path: Path) -> subprocess.Popen:
    command = [
        str(server_path),
        "-m",
        str(model_path),
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
    ]

    if LLAMA_N_GPU_LAYERS:
        command.extend(["-ngl", LLAMA_N_GPU_LAYERS])
    if LLAMA_SERVER_EXTRA_ARGS:
        command.extend(shlex.split(LLAMA_SERVER_EXTRA_ARGS))

    print("Starting llama-server:")
    print("  " + " ".join(shlex.quote(part) for part in command))
    process = subprocess.Popen(command, cwd=ROOT)
    return process


def install_cleanup(process: subprocess.Popen) -> None:
    def cleanup() -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    def handle_signal(signum: int, _frame: object) -> None:
        cleanup()
        raise SystemExit(0)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local PDF parser backend.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Check for and install the latest llama.cpp release before starting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    server_path = ensure_llama_server(update=args.update)
    model_path = ensure_model()

    base_url = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
    os.environ["LLAMA_BASE_URL"] = base_url

    llama_process = start_llama_server(server_path, model_path)
    install_cleanup(llama_process)
    wait_for_llama(base_url)
    print(f"llama-server is ready at {base_url}")

    from app import app

    print(f"Starting backend at http://{BACKEND_HOST}:{BACKEND_PORT}")
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
