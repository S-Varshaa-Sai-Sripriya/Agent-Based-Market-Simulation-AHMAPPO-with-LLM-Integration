from setuptools import setup, find_packages

setup(
    name="AHMAPPO_LLM",
    version="0.1.0",
    description="Machine Learning System for Stock Trading with Sentiment Analysis",
    author="Varshaa Sai Sripriya Saisheshadhri",
    author_email="varshaasaisripriyas@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
        "ta>=0.10.0",
        "gym>=0.26.0",
        "stable-baselines3>=2.0.0",
        "tensorboard>=2.10.0",
        "tqdm>=4.64.0",
        "joblib>=1.1.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)