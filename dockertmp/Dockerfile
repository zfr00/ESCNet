FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV http_proxy=http://192.168.16.5:3128
ENV https_proxy=http://192.168.16.5:3128

RUN PIP_INSTALL="python3 -m pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple" && \
    $PIP_INSTALL transformers==4.9.1 \
    nltk \
    pandas \
    tqdm \
#    csv \
#    re \
    sklearn \
    torchtext==0.7 \
    einops \
#    functools \
#    itertools \
    gensim \
    pillow \
    opencv-python-headless
