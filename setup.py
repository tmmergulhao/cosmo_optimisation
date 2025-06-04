from setuptools import setup, find_packages

setup(
    name="diff_weighted_fields",          # your package name
    version="0.1.0",                      # start with 0.1.0 (or whatever)
    description="1D/ND grid utilities, CIC painting, power‐spectrum tools",
    author="Thiago Mergulhão",
    author_email="thiago.mergulhao@hotmail.com",
    packages=find_packages(where="src"),  # look inside src/ for packages
    package_dir={"": "src"},              # root of code is in src/
    install_requires=[
        "jax",
        "numpy",
    ],
    python_requires=">=3.8",
)