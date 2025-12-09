from setuptools import setup, find_packages

setup(
    name="craft",
    version="0.1.0",
    description="Core Robot Analysis Framework for Tasks",
    author="CRAFT Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "openai>=0.27.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "networkx>=2.5.0",
    ],
)

