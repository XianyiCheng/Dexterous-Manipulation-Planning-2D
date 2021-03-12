
###Dependencies:
    
    sudo apt install python3-pip libglfw3 libeigen3-dev libglfw3-dev git
    sudo apt-get install google-perftools libgoogle-perftools-dev libdw-dev
    pip3 install --user numpy pyopengl pyhull scipy gitpython glfw imgui[full] pyglm matplotlib pycollada sympy pyglm pytest cyglfw3
    pip install qpsolvers

###Locally install contact_modes library:

    cd python 
    pip3 install --user -e .
    
###Locally install itbl library:

    cd python 
    pip3 install --user -e .

###Locally install this library:
    pip install plotly
    pip3 install --user -e .

### install matlab engine for your conda env


    sudo update-alternatives --install /usr/bin/python python ~/anaconda3/envs/py37planning/bin/python 2
    cd "matlabroot\extern\engines\python"
    python setup.py install


### install mpt toolbox for matlab
follow the instructions on https://www.mpt3.org/Main/Installation
