from glob import glob
from setuptools import setup

package = "mobile_panda_bridge"
setup(
    name=package, version="0.1.0", packages=[package],
    data_files=[("share/ament_index/resource_index/packages", [f"resource/{package}"]),
                (f"share/{package}", ["package.xml"]),
                (f"share/{package}/launch", glob("launch/*.py")),
                (f"share/{package}/rviz", glob("rviz/*.rviz"))],
    install_requires=["setuptools"], zip_safe=True,
    entry_points={"console_scripts": ["bridge = mobile_panda_bridge.node:main",
                                      "teleop_check = mobile_panda_bridge.client:main",
                                      "run_skill = mobile_panda_bridge.run_skill:main"]},
)
