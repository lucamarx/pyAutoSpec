FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt clean               && \
    apt update              && \
    apt upgrade -y          && \
    apt install -y locales  && \
    locale-gen en_US.UTF-8

RUN apt install -y             \
    vim                        \
    build-essential            \
    graphviz                   \
    software-properties-common \
    python3                    \
    python3-dev                \
    python3-pip                \
    python3-venv

RUN groupadd -r jupyter && useradd -m -d /workspace -g jupyter jupyter
USER jupyter

RUN python3 -m venv /workspace/venv && \
    /workspace/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /workspace/venv/bin/pip install --no-cache-dir jupyter tabulate

COPY --chown=jupyter . /workspace/pyautospec
RUN cd /workspace/pyautospec && \
    /workspace/venv/bin/pip install -e .

EXPOSE 8888
CMD ["/workspace/venv/bin/jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--notebook-dir=/workspace"]
