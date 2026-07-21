"""Installation script for the Parkour SDK2 sim2sim package."""

from setuptools import find_packages, setup


setup(
    name="parkour_mjlab",
    version="0.0.1",
    packages=find_packages(include=("deploy", "deploy.*")),
    # Asset globs below are explicit; disabling automatic discovery prevents
    # setuptools from treating the XML/mesh directories as Python namespaces.
    include_package_data=False,
    package_data={
        "deploy.parkour.sim2sim": [
            "assets/unitree_g1/LICENSE",
            "assets/unitree_g1/ASSET_SOURCE.md",
            "assets/unitree_g1/*.xml",
            "assets/unitree_g1/meshes/*.STL",
        ]
    },
    install_requires=[
        "mujoco==3.5.0",
        "numpy",
        "onnxruntime",
    ],
)
