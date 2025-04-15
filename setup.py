from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        cur_dir_path = Path.cwd()
        dirs_to_create = [
            cur_dir_path.joinpath('content', 'sample_data', 'Data', 'all_records'),
            cur_dir_path.joinpath('content', 'sample_data', 'Data', 'noise'),
            cur_dir_path.joinpath('content', 'states'),
        ]
        for directory in dirs_to_create:
            if not Path(directory).exists():
                Path(directory).mkdir(parents=True)
                print(f"✅ Создана директория: {directory}")
            else:
                print(f"ℹ️ Директория уже существует: {directory}")


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
