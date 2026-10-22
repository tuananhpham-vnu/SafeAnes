"""Read back saved Kaggle training source; never import or run model code."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/training_source_recovery"
WANTED = {"uc04-w16-v8-tabular", "uc04-w16-v8-tabm"}


def recover() -> None:
    sys.path.insert(0, str(ROOT / ".local_deps/kaggle_client"))
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    credentials = json.loads((ROOT / ".kaggle/tokens.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    found = {}
    for account in credentials:
        api = KaggleApi()
        api.config_values = {api.CONFIG_NAME_USER: account["username"],
                             api.CONFIG_NAME_TOKEN: account["token"],
                             api.CONFIG_NAME_AUTH_METHOD: "ACCESS_TOKEN"}
        try:
            kernels = api.kernels_list(user=account["username"], page_size=100, sort_by="dateRun") or []
        except Exception as error:
            print(json.dumps({"account": account["username"], "error_type": type(error).__name__}), flush=True)
            continue
        matches = [kernel for kernel in kernels if kernel and kernel.ref.rsplit("/", 1)[-1] in WANTED]
        print(json.dumps({"account": account["username"], "matches": [kernel.ref for kernel in matches]}), flush=True)
        for kernel in matches:
            slug = kernel.ref.rsplit("/", 1)[-1]
            request = ApiGetKernelRequest()
            request.user_name, request.kernel_slug = kernel.ref.split("/")
            with api.build_kaggle_client() as client:
                response = client.kernels.kernels_api_client.get_kernel(request)
            source = response.blob.source
            folder = OUT / slug
            folder.mkdir(exist_ok=True)
            (folder / "kaggle_runner.py").write_text(source, encoding="utf-8")
            code = None
            for statement in ast.parse(source).body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "CODE" for target in statement.targets
                ):
                    code = ast.literal_eval(statement.value)
            if not code:
                raise RuntimeError(f"No embedded CODE found in {kernel.ref}")
            archive_bytes = base64.b64decode(code)
            (folder / "training_code.tar.gz").write_bytes(archive_bytes)
            entries = []
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    relative = PurePosixPath(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise RuntimeError("Unsafe archive path")
                    content = archive.extractfile(member).read()
                    destination = folder / "source" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                    entries.append({"path": relative.as_posix(), "size": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()})
            metadata = response.metadata
            evidence = {"kernel": kernel.ref, "version": metadata.current_version_number,
                        "last_run_time": str(metadata.last_run_time), "docker_image": metadata.docker_image,
                        "runner_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(), "files": entries}
            (folder / "recovery_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            found[slug] = {key: value for key, value in evidence.items() if key != "files"}
            print(json.dumps({"recovered": kernel.ref, "version": evidence["version"], "files": len(entries)}), flush=True)
        if WANTED <= found.keys():
            break
    (OUT / "summary.json").write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"recovered": sorted(found), "missing": sorted(WANTED - found.keys())}), flush=True)


if __name__ == "__main__":
    recover()
