## Data pre-processing

Please first download the prepared ABC dataset from [BaiduYun](https://pan.baidu.com/s/1N8E_8xhwKmE2UHuJhdFPZA?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9W7TKeRjZTYydsOW?e=WnVGsE), and unzip it under current folder. We split the models with multiple components. There will be 3 items for each model:

```
*.obj: surface mesh of the model.
*.yml: parametric curve and patch information.
*.fea: curve segments on the mesh. Starts with the number of curve segments n, followed by n lines, where each line contains the two vertex indices of a curve segment. Can be extracted from 'vert_indices' of curves in the *.yml files.
```

**\[Optional\]** If you want to split the models and generate the correponding *.fea files from the raw ABC dataset, please first put the *.yml and *.obj files in folder _abc_data_ (make sure that file in different formats share the same prefix), and run:

        $ python split_and_gen_fea.py

You will find the split models and *.fea files in _raw_input_.

**Installation**

Please first install the Boost and eigen3 library:

        $ sudo apt install libboost-all-dev
        $ sudo apt install libeigen3-dev
Then run:

        $ cd PATH_TO_NH-REP/code/pre_processing
        $ mkdir build && cd build
        $ cmake ..
        $ make

You can generate the training data:

        $ cd ..
        $ python gen_training_data_yaml.py

The generated training data can be found in _training_data_ folder.

If you do not have a yaml file and want to generate sample points from meshes, you can prepare the *.fea file as sharp feature curves of the meshes, then run:

        $ python gen_training_data_mesh.py

Please make sure that you set 'in_path' in _gen_training_data_yaml.py_ and _gen_training_data_mesh.py_ correctly.


When patch decomposition is conducted, there will be *_fixtree.obj and *_fixtree.fea in 'data_path', which can be used for generating point samples in later round.
