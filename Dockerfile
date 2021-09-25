FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.8

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pytorch_lightning && \
    pip install --no-cache-dir tensorboardX && \
    pip install --no-cache-dir matplotlib && \
    pip install --no-cache-dir seaborn && \
    pip install --no-cache-dir torchvision && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir geotorch && \
    pip install --no-cache-dir pyyaml
    
