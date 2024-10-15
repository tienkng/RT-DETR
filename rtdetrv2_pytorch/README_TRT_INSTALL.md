# Instruction install tensorrt 

## Create environment
```bash
conda create -n your-env python==3.10
conda activate your-env
```

## Install with conda
Several of packages get fail when build with `pip`, so let install with `conda` first
```bash
conda install nvidia::cuda-toolkit
conda install pycuda
conda install -c rapidsai -c conda-forge cucim cuda-version=`<CUDA version>`
```
After that, check cuda toolkit has been installed yet.
```
nvcc --version
```

If the installed cuda-toolkit is not successful, you get the notice 'nvcc not found', try the other [solutions](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed)

- Follow the instructions on the [official website](https://pytorch.org/get-started/locally/)

Finally, all the left ones can be installed with `pip`
```bash
pip install -r requirements.txt
pip install -r requirements_trt.txt
```
