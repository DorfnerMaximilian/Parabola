#Add Parabola to python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
file=~/.bashrc
if ! grep -qF "$SCRIPT_DIR" <"$file" ; then echo export PYTHONPATH='${PYTHONPATH}'":$SCRIPT_DIR" >> ~/.bashrc ; fi
#Remove the cache beforehand
rm -r ./__pycache__
rm -r ./Modules/__pycache__
cd ./CPP_Extension/
make clean
cd ..
#compile the CPP extension
cd "./CPP_Extension/"
make
cd ..

