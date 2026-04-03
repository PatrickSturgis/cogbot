from setuptools import setup, find_packages

setup(
    name="cogbot",
    version="0.1.0",
    description="LLM-based cognitive interviewing for survey question pretesting",
    author="Patrick Sturgis",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0",
        "pandas>=2.0",
    ],
    package_data={"cogbot": []},
    data_files=[
        ("data", [
            "data/backstories_short.csv",
            "data/backstories_long.csv",
        ]),
    ],
    include_package_data=True,
)
