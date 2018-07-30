$dir = pwd
$build_dir = "$dir/build"
$cmake_gen = "Visual Studio 15 2017 Win64"

if (Test-Path $build_dir) {
    rm -r $build_dir/*
} else {
    mkdir $build_dir
}

cd $build_dir

cmake -G $cmake_gen ..
# cmake --build . --config release

cd ..