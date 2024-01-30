#!/usr/bin/env python3
import datetime
import json
import signal
import time
from typing import Any, Dict, List

import psutil  # type: ignore[import]
import torch

if not torch.version.hip:
    import pynvml  # type: ignore[import]
else:
    import amdsmi as pyamdsmi  # type: ignore[import]


def get_processes_running_python_tests() -> List[Any]:
    python_processes = []
    for process in psutil.process_iter():
        try:
            if "python" in process.name() and process.cmdline():
                python_processes.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # access denied or the process died
            pass
    return python_processes


def get_per_process_cpu_info() -> List[Dict[str, Any]]:
    processes = get_processes_running_python_tests()
    per_process_info = []
    for p in processes:
        info = {
            "pid": p.pid,
            "cmd": " ".join(p.cmdline()),
            "cpu_percent": p.cpu_percent(),
            "rss_memory": p.memory_info().rss,
        }

        # https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_full_info
        # requires higher user privileges and could throw AccessDenied error, i.e. mac
        try:
            memory_full_info = p.memory_full_info()

            info["uss_memory"] = memory_full_info.uss
            if "pss" in memory_full_info:
                # only availiable in linux
                info["pss_memory"] = memory_full_info.pss

        except psutil.AccessDenied as e:
            # It's ok to skip this
            pass

        per_process_info.append(info)
    return per_process_info


def get_per_process_gpu_info(handle: Any) -> List[Dict[str, Any]]:
    if torch.version.hip is None:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    else:
        processes = pyamdsmi.amdsmi_get_gpu_process_list(handle)
    per_process_info = []
    for p in processes:
        if torch.version.hip:
            proc_info = pyamdsmi.amdsmi_get_gpu_process_info(handle, p)
            info = {
                "pid": proc_info["pid"],
                "gpu_memory": proc_info["memory_usage"]["vram_mem"],
            }
        else:
            info = {"pid": p.pid, "gpu_memory": p.usedGpuMemory}
        per_process_info.append(info)
    return per_process_info


if __name__ == "__main__":
    handle = None
    if not torch.version.hip:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            # no pynvml avaliable, probably because not cuda
            pass
    else:
        try:
            pyamdsmi.amdsmi_init()
            handle = pyamdsmi.amdsmi_get_processor_handles()[0]
        except ModuleNotFoundError:
            pass

    kill_now = False

    def exit_gracefully(*args: Any) -> None:
        global kill_now
        kill_now = True

    signal.signal(signal.SIGTERM, exit_gracefully)

    while not kill_now:
        try:
            stats = {
                "time": datetime.datetime.utcnow().isoformat("T") + "Z",
                "total_cpu_percent": psutil.cpu_percent(),
                "per_process_cpu_info": get_per_process_cpu_info(),
            }
            if handle is not None:
                stats["per_process_gpu_info"] = get_per_process_gpu_info(handle)
                if not torch.version.hip:
                    # https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html
                    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats["total_gpu_utilization"] = gpu_utilization.gpu
                    stats["total_gpu_mem_utilization"] = gpu_utilization.memory
                else:
                    stats["total_gpu_utilization"] = pyamdsmi.amdsmi_get_gpu_activity(handle)["gfx_activity"]
                    stats["total_gpu_mem_utilization"] = pyamdsmi.amdsmi_get_gpu_activity(handle)["umc_activity"]

        except Exception as e:
            stats = {
                "time": datetime.datetime.utcnow().isoformat("T") + "Z",
                "error": str(e),
            }
        finally:
            print(json.dumps(stats))
            time.sleep(1)
