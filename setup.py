from setuptools import find_packages, setup

setup(
    name="anime_lora_finetune",
    version="1.0.0",
    description="Fine-tuning a LoRA model to convert real images into anime-style.",
    packages=find_packages(where="src"),  # Finds packages inside 'src/'
    package_dir={"": "src"},  # Maps all packages to 'src/' directory
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "peft",  # LoRA Fine-Tuning
        "datasets",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "opencv-python",
        "Pillow",
        "tqdm",
        "omegaconf",
        "pytorch_lightning",
        "imageio",
        "scipy",
        "gradio",  # Web UI (if needed)
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],  
    },
    entry_points={
        "console_scripts": [
            "train_lora=scripts.Model_run:train",
            "infer_lora=scripts.Model_run:infer",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
