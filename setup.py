from setuptools import setup, find_packages

setup(
    name="simplifine_alpha",
    version="0.0.7",
    author="Ali Kavoosi, Raveen Kariyawasam, Ege Kan Duman",
    author_email="your.email@example.com",
    description="A brief description of your project",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/simplifine-llm/Simplifine",  
    license="GNU GENERAL PUBLIC LICENSE", 
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
