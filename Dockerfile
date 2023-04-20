FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update     && \
    apt upgrade -y && \
    apt install -y graphviz

COPY . /workspace
WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir jupyter tabulate
RUN pip install --no-cache-dir -e /workspace

EXPOSE 8888
CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--port=8888", "--ip=0.0.0.0",  "--notebook-dir=/workspace"]
