#Add Parabola to python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
file=~/.bashrc
if ! grep -qF "$SCRIPT_DIR" <"$file" ; then echo export PYTHONPATH='${PYTHONPATH}'":$SCRIPT_DIR" >> ~/.bashrc ; fi
#compile the CPP extension
cd "./CPP_Extension/"
make
cd ..

