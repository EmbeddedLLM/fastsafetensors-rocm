# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ROCm/HIP port for AMD GPUs

import os

from setuptools import Extension, setup


def MyExtension(name, sources, mod_name, *args, **kwargs):
    import pybind11

    pybind11_path = os.path.dirname(pybind11.__file__)

    # Detect ROCm installation
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")

    kwargs["define_macros"] = [("__MOD_NAME__", mod_name), ("__HIP_PLATFORM_AMD__", "1")]
    kwargs["libraries"] = ["stdc++", "amdhip64"]
    kwargs["library_dirs"] = [f"{rocm_path}/lib"]
    kwargs["include_dirs"] = kwargs.get("include_dirs", []) + [
        f"{pybind11_path}/include",
        f"{rocm_path}/include",
    ]  # for pybind11/pybind11.h and HIP headers
    kwargs["language"] = "c++"

    # https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
    kwargs["extra_compile_args"] = ["-fvisibility=hidden", "-std=c++17", "-D__HIP_PLATFORM_AMD__"]
    kwargs["extra_link_args"] = [f"-L{rocm_path}/lib", "-lamdhip64"]

    return Extension(name, sources, *args, **kwargs)


setup(
    packages=[
        "fastsafetensors",
        "fastsafetensors.copier",
        "fastsafetensors.cpp",
        "fastsafetensors.frameworks",
    ],
    include_package_data=True,
    package_data={"fastsafetensors.cpp": ["*.hpp", "*.hip.hpp", "cpp.pyi"]},
    ext_modules=[
        MyExtension(
            name=f"fastsafetensors.cpp",
            sources=["fastsafetensors/cpp/ext.hip.cpp"],
            include_dirs=["fastsafetensors/cpp"],
            mod_name="cpp",
        )
    ],
)
