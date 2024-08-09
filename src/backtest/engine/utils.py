# Copyright (C) 2024 Anthony Lavertu
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
        "model": info["model"],
        "family": info["family"] if "family" in info else None,
        "flags": info["flags"] if "flags" in info else None,
        "l2_cache_line_size": info["l2_cache_line_size"] if "l2_cache_line_size" in info else None,
        "l2_cache_associativity": info["l2_cache_associativity"] if "l2_cache_associativity" in info else None,
        "Memory": f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB"
    }