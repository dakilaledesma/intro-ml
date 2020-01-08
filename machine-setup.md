# Machine Setup Guide

Here are the steps that I use to setup machine learning on a machine. My method, however, involves installing Anaconda as a Python interpreter. If you already have a Python installation or preferred way to install Python, my steps would only apply really for Anaconda due to the use of the ```conda``` package manager.

### DISCLAIMER:
Only the following GPUs are supported:

- **NVIDIA**: Check the CUDA capability of your graphics card [here](https://developer.nvidia.com/cuda-gpus). You are good to go if your card has CUDA capability of 3.5 or higher. **Lower than this is not supported by TensorFlow.**
  - Not sure where to look on the website? Check the link and scroll down to "CUDA-Enabled GeForce and TITAN Products"
  - Not sure what Nvidia graphics card you have? 
      - Go to your desktop, right click, and hit "NVIDIA Control Panel." A window should pop up.
      - On the top most toolbar go to Help -> System Information
      - You should see your graphics card in the window as underlined in red in this picture:
      ![GPU_Info](images/gpu_info.png?raw=true "GPU System Information")

## Windows

1. Install the latest version of Anaconda [here](https://www.anaconda.com/distribution/).
2. Once it has installed, open the Anaconda Powershell Prompt by searching it on your start menu.
3. Making an environment for this class is *highly* recommended, but may be skipped:
    - Enter ```conda create -n IntroML python==3.7``` to create a new environment. You will be asked to type ```y/n``` to confirm your changes.
    - Enter ```conda activate IntroML``` to activate the environment
4. Proceed to Nvidia GPUs or AMD GPUs

### Nvidia GPUs
5. Install tensorflow-gpu using ```conda install tensorflow-gpu```
6. Proceed 
 
