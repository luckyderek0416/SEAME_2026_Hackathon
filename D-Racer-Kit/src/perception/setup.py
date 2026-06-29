from setuptools import setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='d-racer team',
    maintainer_email='team@example.com',
    description='OpenCV lane following and ArUco detection nodes',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'lane_node = perception.lane_node:main',
            'aruco_node = perception.aruco_node:main',
        ],
    },
)
