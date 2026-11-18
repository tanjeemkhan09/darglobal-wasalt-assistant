"""Deploy to a Hugging Face Docker Space (free). Usage:
    set HF_TOKEN (write scope) and OPENROUTER_API_KEY in your environment, then
    python scripts/deploy_hf.py [space-name]
HF builds the Dockerfile in the cloud, so no local Docker is needed for the deploy itself.
"""
import os, sys, tempfile, shutil
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1] if len(sys.argv) > 1 else "darglobal-wasalt-assistant"
token, key = os.environ.get("HF_TOKEN"), os.environ.get("OPENROUTER_API_KEY")
if not token or not key:
    sys.exit("Set HF_TOKEN and OPENROUTER_API_KEY first.")

api = HfApi(token=token)
user = api.whoami()["name"]
repo = f"{user}/{name}"
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
api.add_space_secret(repo, "OPENROUTER_API_KEY", key)  # stored as an encrypted Space secret, not in the image

with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    for f in ("Dockerfile", "requirements.txt"):
        shutil.copy(ROOT / f, t / f)
    shutil.copytree(ROOT / "app", t / "app", ignore=shutil.ignore_patterns("__pycache__"))
    (t / "data").mkdir()
    shutil.copytree(ROOT / "data" / "kb", t / "data" / "kb")
    front = "---\ntitle: Dar Global & Wasalt Assistant\nemoji: 🏙️\ncolorFrom: yellow\ncolorTo: green\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n"
    (t / "README.md").write_text(front + (ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(t), commit_message="Deploy assistant")

print("Deployed. Build takes ~2-3 min.")
print(f"Space page: https://huggingface.co/spaces/{repo}")
print(f"Live URL:   https://{user.lower()}-{name.lower()}.hf.space")
