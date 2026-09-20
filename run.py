"""Compile a local .cu file on Modal and run it. GPU is billed only for that job.

    modal run run.py
    modal run run.py --kernel kernel.cu --gpu B200
    modal run run.py --kernel kernel.cu --gpu L4
"""

from pathlib import Path

import modal

# Prebuilt CUDA toolkit image. Built once on CPU, reused forever.
# No pip, no apt — nvcc is already here. Changing kernel.cu does NOT rebuild this.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.12",
    ).entrypoint([])
)

app = modal.App("cu-runner", image=image)

ARCH = {
    "T4": "sm_75",
    "L4": "sm_89",
    "A10": "sm_86",
    "L40S": "sm_89",
    "A100": "sm_80",
    "A100-40GB": "sm_80",
    "A100-80GB": "sm_80",
    "H100": "sm_90a",
    "H100!": "sm_90a",
    "H200": "sm_90a",
    "B200": "sm_100a",
    "B200+": "sm_100a",
    "B300": "sm_100a",
    "RTX-PRO-6000": "sm_120a",
}

SOURCES = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".c"}


@app.function(gpu="H100", timeout=180, scaledown_window=2)
def compile_and_run(
    main: str,
    files: dict[str, str],
    arch: str,
    nvcc_flags: list[str],
    bin_args: list[str],
) -> int:
    import subprocess
    import time
    from pathlib import Path as P

    t0 = time.time()
    work = P("/tmp/src")
    work.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (work / name).write_text(text)

    binary = "/tmp/kernel"
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        str(work / main),
        "-o",
        binary,
        *nvcc_flags,
    ]
    print("$", " ".join(cmd), flush=True)
    built = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if built.stdout:
        print(built.stdout, end="")
    if built.stderr:
        print(built.stderr, end="")
    if built.returncode != 0:
        raise RuntimeError(f"nvcc failed with code {built.returncode}")

    ran = subprocess.run([binary, *bin_args], cwd=work)
    print(f"[modal] wall {time.time() - t0:.2f}s", flush=True)
    if ran.returncode != 0:
        raise RuntimeError(f"kernel exited {ran.returncode}")
    return ran.returncode


@app.local_entrypoint()
def main(
    kernel: str = "kernel.cu",
    gpu: str = "H100",
    nvcc_flags: str = "",
    args: str = "",
):
    path = Path(kernel)
    if not path.is_file():
        raise SystemExit(f"not found: {path}")
    if gpu not in ARCH:
        raise SystemExit(f"unknown gpu {gpu!r}. pick one of: {', '.join(ARCH)}")

    files = {
        p.name: p.read_text(encoding="utf-8")
        for p in path.parent.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCES
    }
    extra = nvcc_flags.split() if nvcc_flags.strip() else []
    bin_args = args.split() if args.strip() else []

    print(f"running {path.name} on {gpu}  arch={ARCH[gpu]}  files={sorted(files)}")
    compile_and_run.with_options(gpu=gpu).remote(
        path.name, files, ARCH[gpu], extra, bin_args
    )
