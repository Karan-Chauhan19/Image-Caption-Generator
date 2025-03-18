'''
author: Karan Chauhan
github: @Karan-Chauhan19
organization: L.J University
'''

import torch
import subprocess

def check_gpu() -> None:
    """
    This function is used to check, whether GPUs are available.

    Parameters:
    - ()

    Returns:
    - (None)
    """

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count() # get total number of available gpus
        print("- Number of GPUs available: {}".format(num_gpus))

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name() # get gpu name
            print("- GPU name: {}".format(gpu_name))

        command = "nvidia-smi" # set a command
        result = subprocess.run(command, shell=True, capture_output=True, text=True) # execute a command
        
        if result.returncode == 0:
            print(result.stdout) # output after successful execution of command
        else:
            print("- Error message: \n{}".format(result.stderr)) # output after failed exection of command
    
    else:
        print("- CUDA is not available. Using CPU instead.")