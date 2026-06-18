
pip install -r requirements.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html

cd step1x3d_texture/custom_rasterizer
python3 setup.py install
cd ../differentiable_renderer
python3 setup.py install
cd ../../

pip install --upgrade transformers
pip install --upgrade diffusers