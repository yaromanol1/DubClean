"""Safe non-interactive stop helper used by stop.bat."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.storage import Store, utc_now  # noqa: E402


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def process_command_line(pid: int) -> str | None:
    """Command line of a live process, or None when it cannot be determined."""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def pid_matches(pid: int | None, *markers: str) -> bool | None:
    """True/False when the process identity is known; None when unknown.

    PIDs are recycled by Windows (especially across reboots), so a stored PID
    must never be killed without confirming the command line still belongs to
    this module — otherwise stop.bat could terminate an unrelated program.
    """
    if not pid_alive(pid):
        return False
    command_line = process_command_line(int(pid))
    if command_line is None:
        return None
    return all(marker.lower() in command_line.lower() for marker in markers)


def kill_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def kill_verified(pid: int | None, *markers: str) -> bool:
    """Kill the process tree only when its command line matches the markers."""
    if not pid:
        return False
    verdict = pid_matches(int(pid), *markers)
    if verdict:
        kill_tree(int(pid))
        return True
    return False


def find_module_instances(*markers: str) -> list[int]:
    """PIDs of live python processes whose command line matches all markers.

    Needed because a PID file can be lost while the panel itself keeps
    running (and Windows allows a second bind to the same port via
    SO_REUSEADDR, silently splitting traffic between two instances).
    """
    condition = " -and ".join(
        f"$_.CommandLine -like '*{marker}*'" for marker in markers
    )
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or "
                "Name='pythonw.exe'\" | Where-Object { $_.CommandLine -and "
                f"({condition}) }} | ForEach-Object {{ $_.ProcessId }}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    pids = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    # The venv python.exe on this machine is a launcher shim: it spawns the
    # real interpreter as a child with an identical command line, so every
    # instance shows up as two PIDs. Exclude the caller and its parent, or
    # the panel would mistake its own launcher for a second instance.
    own = {os.getpid(), os.getppid()}
    return [pid for pid in pids if pid not in own]


def stop_running_tasks(store: Store, reason: str) -> int:
    """Stop every queued or running task and record why it ended.

    Shared by stop.bat and the panel's own shutdown button so that closing the
    service one way cannot leave workers behind that the other way would have
    killed.  Returns how many tasks were stopped.
    """
    stopped = 0
    for task in store.list_tasks(states={"queued", "starting", "running"}):
        Path(task["stop_file"]).write_text("stop", encoding="utf-8")
        kill_verified(task.get("pid"), "task_worker.py", str(task["id"]))
        task.update(
            {
                "state": "stopped",
                "stage": "Остановлено",
                "substage": reason,
                "finished_at": utc_now(),
                "heartbeat_at": utc_now(),
                "eta_seconds": None,
                "pid": None,
                "error": None,
            }
        )
        store.save_task(task)
        stopped += 1
    return stopped


def stop_all() -> int:
    store = Store()
    stop_running_tasks(store, "Сервис остановлен через stop.bat.")
    pid_file = store.runtime_root / "web.pid"
    exit_code = 0
    recorded_pid: int | None = None
    if pid_file.is_file():
        try:
            recorded_pid = int(pid_file.read_text(encoding="ascii").strip())
        except ValueError:
            print("Повреждён PID-файл; файл удалён, посторонние процессы не трогались.")
            exit_code = 1
        pid_file.unlink(missing_ok=True)
    if recorded_pid is not None and pid_alive(recorded_pid):
        if kill_verified(recorded_pid, "paired_reference_cancel", "app.py"):
            print(f"Сервис (PID {recorded_pid}) остановлен.")
        else:
            print(
                f"PID {recorded_pid} занят посторонним процессом (переиспользован "
                "системой); он НЕ остановлен."
            )
    # A lost/overwritten PID file must not leave orphaned panels running:
    # every remaining verified instance of this module's app.py is stopped too.
    stray = find_module_instances("paired_reference_cancel", "app.py")
    for pid in stray:
        if kill_verified(pid, "paired_reference_cancel", "app.py"):
            print(f"Остановлен дополнительный экземпляр панели (PID {pid}).")
    if recorded_pid is None and not stray:
        print("Сервис не запущен.")
    else:
        print("Остановка завершена.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(stop_all())
