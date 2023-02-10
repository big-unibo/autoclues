FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.6
COPY . /home/autoclues
WORKDIR /home/autoclues
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN mkdir results
RUN chmod 777 scripts/*
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]