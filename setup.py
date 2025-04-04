import os
from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        dirs_to_create = [
            os.path.join(os.getcwd(), 'content'),
            os.path.join(os.getcwd(), 'content', 'sample_data'),
            os.path.join(os.getcwd(), 'content', 'sample_data', 'Data'),
            os.path.join(os.getcwd(), 'content', 'sample_data', 'Data', 'all_records'),
            os.path.join(os.getcwd(), 'content', 'sample_data', 'Data', 'noise'),
            os.path.join(os.getcwd(), 'content', 'states'),
        ]
        for directory in dirs_to_create:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Создана директория: {directory}")
            else:
                print(f"Директория уже существует: {directory}")

# Загружаем зависимости из requirements.txt
with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="music_mmls",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Margarita&Artem&Matvey",
    description="Denoising for music",
    url="https://github.com/Kovylyaev/Music_MMLS",
    cmdclass={
        'install': PostInstallCommand,
    },
)
