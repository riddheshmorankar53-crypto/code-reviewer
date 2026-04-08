from setuptools import setup, find_packages

setup(
    name="codereviewenv",
    version="0.1.0",
    description="OpenEnv environment for pull-request code review with deterministic grading",
    author="OpenEnv Community",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "pydantic>=2.6,<3.0",
        "openai>=1.30,<2.0",
        "fastapi>=0.115,<1.0",
        "uvicorn>=0.30,<1.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "pytest-asyncio>=0.23", "ruff>=0.4", "mypy>=1.9"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Quality Assurance",
    ],
)
