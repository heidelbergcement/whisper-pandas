from setuptools import setup

setup(
    name="whisper-pandas",
    version="0.1",
    install_requires=["pandas", "whisper"],
    description="WhisperDB Python Pandas Reader",
    author="Christoph Deil",
    author_email="Deil.Christoph@gmail.com",
    url="https://github.com/heidelbergcement/whisper-pandas/",
    license="MIT",
    py_modules=["whisper_pandas"],
    entry_points={
        "console_scripts": ['whisper-pandas=whisper_pandas:main'],
    }
)
