ARG BASE_CONTAINER=jupyter/minimal-notebook:latest
FROM $BASE_CONTAINER

USER $NB_USER


RUN export NODE_OPTIONS=--max-old-space-size=4096 \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build \
    && jupyter labextension install @krassowski/jupyterlab-lsp \
    && jupyter lab build \
    && unset NODE_OPTIONS


##Install VS Code LSP
RUN pip install jupyter-lsp \
    && pip install python-language-server[all]

RUN pip install numpy pandas sklearn tpot auto-sklearn