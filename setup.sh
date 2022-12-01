# create conda environment
conda env remove -n lkirstenisbi
conda env create -f environment.yml

# activate conda env
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate lkirstenisbi

# create symbolic link
echo "Creating symbolic link to /workdir"
ln -s $(pwd)/.. /workdir

# give read and write permitions to /workdir
#sudo chmod -R 777 /workdir

# build cuda dependecies
cd /workdir/SW/RotationDetection/libs/utils/cython_utils
rm *.so
rm *.c
rm *.cpp
python setup.py build_ext --inplace

cd /workdir/SW/RotationDetection/libs/utils
rm *.so
rm *.c
rm *.cpp
python setup.py build_ext --inplace
