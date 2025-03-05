from setuptools import setup, find_packages

setup(
    name="data_annotator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'matplotlib',
        'keyboard'
    ],
    author="Viktor Shitov",
    author_email="viktovdmit@yandex.ru",
    description="A tool for annotating bounding boxes and keypoints in images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ankluz/data_annotator",
    project_urls={
        "Bug Tracker": "https://github.com/Ankluz/data_annotator/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="annotation, bounding-box, keypoints, computer-vision, image-processing",
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        "": ["LICENSE", "NOTICE"],
    },
) 