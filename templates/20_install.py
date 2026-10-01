# Install the locked library versions (Kaggle only; the local venv is the lock).
def sh(*cmd):
    """Run a command and stream its output into the notebook; raise on failure."""
    print("$", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise RuntimeError(f"command failed with exit code {proc.returncode}")


if RUNTIME == "kaggle":
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", REPO / "requirements-lock.txt")
    if NEEDS_TORCH:
        torch_version = json.loads((REPO / "requirements-lock.json").read_text())["torch_base_version"]
        sh(sys.executable, "-m", "pip", "install", "-q", f"torch=={torch_version}", "--index-url", TORCH_INDEX)
    sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", REPO)
