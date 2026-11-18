from setuptools import find_packages, setup

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'optax'],
    zip_safe=True,
    maintainer='wenju',
    maintainer_email='lmourelatos@tudelft.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller = controller.control_input:main',
            'static_controller = controller.static_control_input:main',
            'dummy = controller.dummy_node:main',
            'online_controller = controller.static_online_controller:main',
            'online_scratch_controller = controller.static_online_scratch_controller:main',
            'online_finetuning = controller.static_online_finetuning:main',
            'online_lora = controller.static_online_lora:main',
            'online_oneshot = controller.static_online_oneshot:main',
        ],
    },
)
