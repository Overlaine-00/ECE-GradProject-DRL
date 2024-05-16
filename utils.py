import os



def is_in_docker() -> bool:
    """
    Check if the code is running in a Docker container
    """
    return os.path.exists('/.dockerenv')


in_docker = is_in_docker()
def get_path(path: str) -> str:
    if not in_docker:
        return path.replace('/', '\\')
    return path


# def get_full_path(path: str) -> str:
#     '''
#     path: relative path, starting from grad_project or workspace
#         should not include ./ or .\\
        
#     Returns: full path
#     '''
#     parent_dir = os.getcwd()
#     if in_docker:
#         return parent_dir + path
#     else:
#         return parent_dir + path.replace('/', '\\')