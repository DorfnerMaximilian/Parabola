#Add Parabola to python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo export PYTHONPATH='${PYTHONPATH}'":$SCRIPT_DIR" >> ~/.bashrc
#compile the CPP extension
cd "./CPP_Extension/"
make
cd ..

