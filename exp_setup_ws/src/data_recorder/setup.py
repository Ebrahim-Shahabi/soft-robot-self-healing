from setuptools import find_packages, setup

package_name = 'data_recorder'

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
    maintainer='wenju',
    maintainer_email='lmourelatos@tudelft.nl',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'online_data_recorder = data_recorder.online_recorder_function:main',
                'annotated_video = data_recorder.annotated_video:main',
                'event_recorder = data_recorder.event_recorder_function:main',
                'event_annotated_video = data_recorder.event_annotated_video:main',
                'syncer = data_recorder.event_apriltag_syncer:main',

        ],
},
    
)
