# What this notebook produced.
files = sorted(p for p in WORK.rglob("*") if p.is_file())
print(f"{len(files)} files in {WORK}")
for p in files[:40]:
    print("  ", p.relative_to(WORK))
if len(files) > 40:
    print(f"   ... and {len(files) - 40} more")
