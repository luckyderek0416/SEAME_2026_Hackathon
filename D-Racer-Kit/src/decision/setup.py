import os
from glob import glob
from setuptools import setup

package_name = 'decision'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='d-racer team',
    maintainer_email='team@example.com',
    description='State-machine decision node (lane PID + missions) for D-Racer',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'decision_node = decision.decision_node:main',
        ],
    },
)
