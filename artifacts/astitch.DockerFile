FROM bladedisc/bladedisc:latest-runtime-torch1.7.1-cu110

RUN pip install einops timm transformers

COPY . scripts/

WORKDIR scripts
