import sys
import platform
import cpuinfo
import psutil

def get_py_version() -> str:
    """
    This method return the python version as a string.
    :return: Python version as a string
    """
    return ".".join([str(e) for e in sys.version_info[:3]])


def get_platform() -> dict:
    """
    This method return the platform information.
    :return: platform info as a dictionary
    """
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }


def get_hardware() -> dict:
    """
    This method return the hardware information.
    :return: hardware info as a dictionary
    """
    info = cpuinfo.get_cpu_info()
    return {
        "arch": info["arch_string_raw"],
        "bits": info["bits"],
        "brand": info["brand_raw"],
        "count": info["count"],
        "hz_advertised": info["hz_advertised_friendly"],
        "hz_actual": info["hz_actual_friendly"],
        "vendor_id": info["vendor_id_raw"],
        "l2_cache_size": info["l2_cache_size"],
        "stepping": info["stepping"],
        "model": info["model"],
        "family": info["family"],
        "flags": info["flags"],
        "l2_cache_line_size": info["l2_cache_line_size"],
        "l2_cache_associativity": info["l2_cache_associativity"],
        "Memory": f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB"
    }