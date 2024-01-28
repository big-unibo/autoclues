FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.6
RUN cd home && mkdir dump
WORKDIR /home/dump
COPY datasets datasets
COPY scenarios scenarios
COPY experiment experiment
COPY resources resources
COPY scripts scripts
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN mkdir results
RUN chmod 777 scripts/*
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]