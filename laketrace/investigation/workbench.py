"""Recorded scientific computation on an explicit measured-data snapshot.

macOS Seatbelt provides the filesystem/network/process boundary. No unsandboxed
fallback is permitted. Replay verifies recorded inputs and artifacts instead of
rerunning arbitrary, potentially nondeterministic model-written programs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import site
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from laketrace.util import dump_json, sha256_file, stable_hash


def sandbox_profile(work):
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
        raise RuntimeError("Scientific workbench requires macOS Seatbelt; no unsafe fallback")
    executable = Path(sys.executable).resolve()
    reads = {Path(sys.base_prefix).resolve(), executable.parent,
             Path(sys.executable).parent.resolve(), Path(sys.prefix) / "pyvenv.cfg",
             *[Path(p).resolve() for p in site.getsitepackages()],
             Path("/System"), Path("/usr/lib"), Path("/usr/share"), Path("/dev"), work}
    profile = ['(version 1)', '(deny default)', '(allow sysctl-read)',
               '(allow mach-lookup)', '(allow file-read-metadata)',
               '(allow file-read* (literal "/"))',
               '(allow process-exec (literal ' + json.dumps(str(executable)) + '))']
    profile += ['(allow file-read* (subpath ' + json.dumps(str(p)) + '))' for p in sorted(reads)]
    profile += ['(allow file-write* (subpath ' + json.dumps(str(work)) + '))']
    return "\n".join(profile)


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, 2 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def compute(code, data, directory, index, *, replay=False, api=1):
    if not isinstance(code, str) or not code.strip() or len(code) > 16000:
        raise ValueError("code must contain 1-16000 characters of Python")
    directory = Path(directory) / f"analysis-{index:03d}"
    inputs = {"code": code, "data": data}
    if api >= 2:
        inputs['workbench_api'] = api
    signature = stable_hash(inputs)
    receipt = directory / "receipt.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["input_sha256"] != signature:
            raise ValueError("analysis replay input mismatch")
        for name, digest in saved["files"].items():
            path = directory / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("analysis artifact integrity mismatch")
        return saved["result"]
    if replay:
        raise ValueError("missing recorded analysis; replay never executes model code")
    directory.mkdir(parents=True, exist_ok=False)
    dump_json(directory / "input.json", data)
    (directory / "analysis.py").write_text(code)
    started = time.monotonic()
    if api >= 3:
        try:
            compile(code, 'analysis.py', 'exec')
        except SyntaxError as exc:
            result = {'ok':False, 'analysis_id':index, 'input_sha256':signature,
                'observation_samples':len(data['samples']), 'stdout':'', 'stderr':str(exc),
                'termination':'syntax_error', 'returncode':None, 'artifacts':[], 'truncated':False,
                'hint':'Provide valid Python with actual newlines, not double-escaped newline sequences. No code executed; use inspect for standard statistics.'}
            dump_json(receipt, {'input_sha256':signature, 'result':result,
                'files':{p.name:sha256_file(p) for p in directory.iterdir() if p.is_file()},
                'wall_time_s':time.monotonic()-started, 'isolation':'syntax validation only; no execution'})
            return json.loads(receipt.read_text())['result']
    with tempfile.TemporaryDirectory(prefix="laketrace-science-") as temporary:
        work = Path(temporary).resolve()
        profile = sandbox_profile(work) if sys.platform == "darwin" else None
        if profile is None and (api < 3 or sys.platform != "linux" or not shutil.which("bwrap")):
            raise RuntimeError("Scientific execution requires macOS Seatbelt or Linux bubblewrap; no unsafe fallback")
        (work / "input.json").write_text(json.dumps(data, allow_nan=False))
        (work / "analysis.py").write_text(code)
        bootstrap = "import json, random, numpy as np\nrandom.seed(0)\nnp.random.seed(0)\ndata=json.load(open('input.json'))\nexec(compile(open('analysis.py').read(), 'analysis.py', 'exec'))\n"
        if api >= 2:
            bootstrap = bootstrap.replace("exec(compile", "import pandas as pd\ndf=pd.DataFrame(data.get('target_nodes', []))\nrows=df.to_dict('records')\nexec(compile")
        if api >= 3:
            bootstrap = bootstrap.replace('exec(compile', "import time, os\nopen('.ready.tmp','w').write(str(time.monotonic()))\nos.replace('.ready.tmp','.ready')\nexec(compile")
        (work / "run.py").write_text(bootstrap)
        environment = {"PATH": "/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work),
                       "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(work / ".mpl"),
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                       "VECLIB_MAXIMUM_THREADS": "1"}
        with (work / "stdout.txt").open("wb") as stdout, (work / "stderr.txt").open("wb") as stderr:
            command = [sys.executable, '-I', '-B', str(work/'run.py')]
            if profile is not None:
                command = ['/usr/bin/sandbox-exec', '-p', profile, *command]
            else:
                prefix = [shutil.which('bwrap'), '--unshare-all', '--die-with-parent', '--new-session', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
                for path in sorted({Path('/usr'), Path('/lib'), Path('/lib64'), Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()}):
                    if path.exists():
                        prefix += ['--ro-bind', str(path), str(path)]
                command = [*prefix, '--bind', str(work), str(work), '--chdir', str(work), *command]
            process = subprocess.Popen(command,
                cwd=work, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, preexec_fn=_limits)
            deadline = time.monotonic() + (30 if api >= 3 else 15)
            ready_at = None
            termination = None
            while True:
                try:
                    returncode = process.wait(timeout=.25)
                    break
                except subprocess.TimeoutExpired:
                    if api >= 3 and ready_at is None and (work/'.ready').exists():
                        ready_at = float((work/'.ready').read_text())
                        deadline = ready_at + 15
                    rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                                         capture_output=True, text=True).stdout.strip()
                    if time.monotonic() >= deadline or (rss.isdigit() and int(rss) > 1024**2):
                        termination = "memory_limit" if rss.isdigit() and int(rss)>1024**2 else "startup_timeout" if api>=3 and ready_at is None else "wall_timeout"
                        process.kill()
                        process.wait()
                        returncode = -1
                        break
        # Execute the venv interpreter path: its pyvenv.cfg selects dependencies.
        artifacts = []
        for path in sorted(work.iterdir()):
            if path.name in {"input.json", "analysis.py", "run.py"} or path.is_symlink() or not path.is_file():
                continue
            if path.suffix not in {".png", ".json", ".csv", ".txt"} or path.stat().st_size > 2 * 1024**2:
                continue
            if len(artifacts) >= 10:
                break
            (directory / path.name).write_bytes(path.read_bytes())
            artifacts.append(path.name)
        stdout = (work / "stdout.txt").read_bytes()
        stderr = (work / "stderr.txt").read_bytes()
        result = {"ok": returncode == 0, "analysis_id": index,
                  "input_sha256": signature, "observation_samples": len(data["samples"]),
                  "stdout": stdout[:8000].decode(errors="replace"),
                  "stderr": stderr[-3000:].decode(errors="replace"),
                  "truncated": len(stdout) > 8000 or len(stderr) > 3000,
                  "returncode": returncode, "artifacts": artifacts,
                  "scope": "Model-written computation on measured data; predictions are not observations. PNG artifacts are saved for inspection, not automatically seen by the model."}
        if api >= 3:
            if ready_at is None and (work/'.ready').exists():
                ready_at = float((work/'.ready').read_text())
            result['termination'] = termination or ('success' if returncode == 0 else 'cpu_limit' if returncode == -signal.SIGXCPU else 'process_signal_'+str(-returncode) if returncode<0 else 'initialization_error' if not (work/'.ready').exists() else 'runtime_error')
            result['startup_wall_s'] = None if ready_at is None else ready_at-started
            result['hint'] = 'Use inspect for standard statistics; inspect(view=artifact,analysis_id,artifact) reads recorded text/CSV/JSON. Each code call starts fresh. No failed action is assumed to have run.'
        if api >= 2:
            result['dataframe_columns'] = sorted(data.get('target_nodes',[{}])[0]) if data.get('target_nodes') else []
            if returncode != 0:
                result['hint'] = "Fresh process: df is the measured target_nodes DataFrame, already joined to background_median/lod; rows=df.to_dict('records'); data['calibration'] is a list of dicts. Example: print(df.nlargest(5,'target_value')[['node_id','x','y','z','target_value','background_median','lod']].to_string(index=False)). Do not assume variables from previous calls exist."
    files = {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}
    dump_json(receipt, {"input_sha256": signature, "result": result, "files": files,
                        "wall_time_s": time.monotonic() - started,
                        "limits": {"cpu_s": 10, "wall_s": 15, "startup_wall_s": 30 if api>=3 else None, "rss_watchdog_bytes": 1024**3},
                        "isolation": "macOS Seatbelt: allowlisted runtime/observations; no network or fork" if sys.platform=="darwin" else "Linux bubblewrap: unshared namespaces, readonly runtime, no network; process-count limit 64"})
    return json.loads(receipt.read_text())["result"]
