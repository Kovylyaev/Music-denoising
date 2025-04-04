import os

def listdir_fullpath(directory):
    """
    Возвращает список полных путей для всех файлов в указанной директории.
    
    Args:
        directory (str): Путь к директории.
        
    Returns:
        list: Список путей к файлам.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory)]
