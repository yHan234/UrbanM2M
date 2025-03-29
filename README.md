# UrbanM2M

## **Development environment configuration**

**Note**: The following process has been validated in both Windows and Centos7, but haven't been validated in MacOS yet.

1. install CUDA 11.3 and cuDNN 8.2.1 from NVIDIA.com. Details of installing CUDA and cuDNN can be found in https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows .
  
   **Note**: to install  CUDA and cuDNN, you must have a NVIDIA GPU on your device.
2. install a [Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) or [Anaconda](https://www.anaconda.com/) package manager.

3. run the following commands in cmd or bash to create a new virtual environment for UrbanM2M.

``` bash
conda create -n urbanm2m python==3.10.0
```

4. install PyTorch

``` bash
conda activate urbanm2m
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
```

5.  install GDAL-Python(version>=3.0.0)

If you are using Linux, run 
```bash 
conda install gdal
``` 
directly to install it.

If you are using Windows, it is better to install GDAL-Python using the .whl file in the m2mCode folder.
``` bash
cd m2mCode
pip install GDAL-3.4.3-cp310-cp310-win_amd64.whl
``` 


6. run 
   
```bash
pip install requirements.txt
``` 
to install other dependency packages.

**Note**: please ensure all the procedures above are run in the **urbanm2m** Virtual Environment

## **UrbanM2M model implementation**

### **Generating raster tiles**

Run the following command to generate raster tiles in GBA.
``` bash
python split.py
```
You can also generate raster tiles for other regions after changing the ```data_root_dir``` variable in the code.

**Note**: this process would be slow if data is on HHD disk, and very faster if data is on SSD disk.

By running split.py, two folders storaging tiles will be generated in the ```data-gisa-gba``` folder

### **Training your model**

Run ```train.py``` using an IDE directly or run the following command in cmd or bash (**recommended**).
```bash
python train.py \
   --start_year 2000 \
   --enc_len 6 \
   --fore_len 6 \
   --height 64 \
   --block_dir ../hzb/block64_64 \
   --nlayers 2 \
   --filter_size 5 \
   --epochs 60 \
   --batch_size 8 \
   --lr 0.00005 \
   --eta_decay 0.015 \
   --sample_count 5000 \
   --val_prop 0.25 \
   --model_type hzb
```

**Note**: the parameters are modifiable. Especially, check your GPU memory to set ```batch_size```. Per batch size needs about 1.3GB GPU memory when training.

**Note**: it is recommended to end training after at least 15 epochs. The trained models will be storaged in ```trained_models``` folder.

### **Testing your model**

After finishing training, modify the parameters and run ```train.py``` using an IDE directly or run the following command in cmd or bash (**recommended**).

``` bash
python test.py \
   --start_year 2006 \
   --enc_len 6 \
   --fore_len 6 \
   --height 64 \
   --block_step 38 \
   --edge_width 4 \
   --model_type gba \
   --region gba \
   --data_root_dir ../data-gisa-gba \
   --log_file ./mylog/gbamodel-gba.csv \
   --model_path ./trained_models/gba-fs5-variable7-p.pth \
   --run_model True \
   --numworkers 0 \
   --batch_size 100
```

Aftering finish testing, results can be found in ```data-gisa-gba/sim``` folder.

**Note**: the parameter model_path must be modified according to name of your trained model.  

**Note**: check your GPU memory to set ```batch_size```. Per batch size needs about 40MB GPU memory when testing.

**Note**: a trained model has been prepared in the ```trained_models``` folder, you can directly test this model without training.

### **Transfering your model**

Before transfering the trained model, you should generate raster tiles for BTT and YRD by running ```split.py``` first.

To transfer GBA model in BTT:

``` bash
python test.py \
   --start_year 2006 \
   --enc_len 6 \
   --fore_len 6 \
   --height 64 \
   --block_step 38 \
   --edge_width 4 \
   --model_type gba \
   --region gba \
   --data_root_dir ../data-gisa-gba \
   --log_file ./mylog/gbamodel-yrd.csv \
   --model_path ./trained_models/gba-fs5-variable7-p.pth \
   --run_model True \
   --numworkers 0 \
   --batch_size 100
```

To transfer GBA model in YRD:

``` bash
python test.py \
   --start_year 2006 \
   --enc_len 6 \
   --fore_len 6 \
   --height 64 \
   --block_step 38 \
   --edge_width 4 \
   --model_type gba \
   --region yrd \
   --data_root_dir ../data-gisa-gba \
   --log_file ./mylog/gbamodel-yrd.csv \
   --model_path ./trained_models/gba-fs5-variable7-p.pth \
   --run_model True \
   --numworkers 0 \
   --batch_size 100
```