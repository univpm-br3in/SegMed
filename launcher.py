"""Entry point for the installed Windows application."""
import ctypes
import os
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path


def show_error(message):
    ctypes.windll.user32.MessageBoxW(None, message, "SegMed", 0x10)


def main():
    app_dir = Path(__file__).resolve().parent
    log_dir = Path(os.environ.get("LOCALAPPDATA", str(app_dir))) / "SegMed" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "launch.log"

    if not (app_dir / ".installation-complete").exists():
        show_error(
            "The SegMed inference environment is not configured.\n\n"
            "Run 'Repair inference environment' from the SegMed Start menu folder."
        )
        return 1

    # GPU UUIDs remain stable when a machine contains multiple NVIDIA devices.
    device_file = app_dir / ".cuda-device-uuid"
    if device_file.is_file():
        os.environ["CUDA_VISIBLE_DEVICES"] = device_file.read_text(encoding="ascii").strip()
    os.chdir(app_dir)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] SegMed launch\n")
            original_stdout, original_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = stream
                sys.stderr = stream
                runpy.run_path(str(app_dir / "main.py"), run_name="__main__")
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr
    except Exception:
        with log_path.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        show_error(f"SegMed could not start.\n\nDetails: {log_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
