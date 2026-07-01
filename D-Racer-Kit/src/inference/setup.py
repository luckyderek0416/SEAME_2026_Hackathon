from setuptools import setup

package_name = 'inference'

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
    description='YOLO object detection node (traffic lights + turn signs)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'yolo_node = inference.yolo_node:main',
            'yolo_ncnn_node = inference.yolo_ncnn_node:main',
            'detection_viz_node = inference.detection_viz_node:main',
        ],
    },
)
