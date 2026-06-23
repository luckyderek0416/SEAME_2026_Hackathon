from setuptools import find_packages, setup

package_name = 'makedb'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taehulim',
    maintainer_email='taehulim@example.com',
    description='Convert D-Racer rosbag topics into image steering dataset',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'makedb_node = makedb.makedb_node:main',
        ],
    },
)
