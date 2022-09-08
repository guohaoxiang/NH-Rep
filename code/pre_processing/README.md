## Data pre-processing

Please first download the prepared ABC dataset from [BaiduYun](https://pan.baidu.com/s/1N8E_8xhwKmE2UHuJhdFPZA?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9W7TKeRjZTYydsOW?e=WnVGsE), and unzip it under current folder. We split the models with multiple components. There will be 3 items for each model:

```
*.obj: surface mesh of the model.
*.yml: parametric curve and patch information.
*.fea: curve segments on the mesh. Starts with the number of curve segments n, followed by n lines, where each line contains the two vertex indices of a curve segment. Can be extracted from 'vert_indices' of curves in the *.yml files.
```

**\[Optional\]** If you want to split the models and generate the correponding *.fea files from the raw ABC dataset, please first put the *.yml and *.obj files in folder _abc_data_ (make sure that file in different formats share the same prefix). Install the PyYAML package via:

        $ pip install PyYAML        

and run:

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

Please make sure that you set 'in_path' in _gen_training_data_yaml.py_ and _gen_training_data_mesh.py_ as the path containing the *.fea files.


When patch decomposition is conducted (like model 00007974_5), there will be *_fixtree.obj and *_fixtree.fea in _training_data_, which can be used for generating point samples in later round:

        $ python gen_training_data_yaml.py -r

or 

        $ python gen_training_data_mesh.py -r
        
You can find the generated training data of the decomposed patch in _training_data_repair_. By default we only decompose one patch and it is enough for most models. But if you find *_fixtree.obj and *_fixtree.fea in _training_data_repair_, that means that more patches need to decomposed. There are two ways to achieve this. First, you can copy _training_data_repair./*_fixtree.obj_ and _training_data_repair./*_fixtree.fea_ to _training_data_repair_, and re-run 'python gen_training_data_yaml.py -r', until enough patches are decomposed (*.conf files can be found in _training_data_repair_). Another way is to decompose all patches at once, to achieve this, simple uncomment the following line in _FeatureSample/helper.cpp_:

https://github.com/guohaoxiang/NH-Rep/blob/42ae22bf8fc3f1b4f9f5592443c29aafe86905bd/code/pre_processing/FeatureSample/helper.cpp#L722

and re-run 'python gen_training_data_yaml.py' and 'python gen_training_data_yaml.py -r'. There will be generated training data in _training_data_repair_.
