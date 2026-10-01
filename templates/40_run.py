# Run the step. Each script writes only its own files and skips work already done.
def fill(arg):
    return {"{SAMPLES}": SAMPLES, "{WORK}": WORK, "{PREP}": PREP}.get(arg, arg)


for step in RUN:
    args = [fill(a) for a in step]
    if "{INPUTS}" in args:
        i = args.index("{INPUTS}")
        args = args[:i] + [str(p) for p in INPUTS] + args[i + 1:]
    sh(sys.executable, REPO / args[0], *args[1:], *EXTRA)
