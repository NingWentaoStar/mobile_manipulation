import os
from glob import glob
from setuptools import setup

package = "mobile_panda_description"
data = [("share/ament_index/resource_index/packages", [f"resource/{package}"]),
        (f"share/{package}", ["package.xml"]),
        (f"share/{package}/urdf", glob("urdf/*.urdf")),
        (f"share/{package}/launch", glob("launch/*.py"))]
for directory, _, files in os.walk("meshes"):
    if files:
        data.append((f"share/{package}/{directory}", [os.path.join(directory, f) for f in files]))
setup(name=package, version="0.1.0", packages=[], data_files=data, install_requires=["setuptools"], zip_safe=True)
