# Library versions must match requirements-lock.json (model-critical packages).
sh(sys.executable, REPO / "scripts" / "env_check.py", *([] if NEEDS_TORCH else ["--ignore", "torch"]))
if NEEDS_TORCH:
    r = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available(), "
                        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                       capture_output=True, text=True)
    print("torch:", r.stdout.strip() or r.stderr.strip())
