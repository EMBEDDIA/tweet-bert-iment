FROM python:3.9-slim

# Install tini and create an unprivileged user
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini /sbin/tini
RUN addgroup --gid 1001 "elg" && adduser --disabled-password --gecos "ELG User,,," --home /elg --ingroup elg --uid 1001 elg && chmod +x /sbin/tini
#RUN apt-get install curl source

# Copy in just the requirements file
COPY --chown=elg:elg requirements.txt /elg/

# Everything from here down runs as the unprivileged user account
USER elg:elg

WORKDIR /elg


# Create a Python virtual environment for the dependencies
RUN python -m venv venv 
RUN /elg/venv/bin/python -m pip install --upgrade pip
RUN venv/bin/pip install tokenizers
RUN venv/bin/pip --no-cache-dir install -r requirements.txt 

# Copy ini the entrypoint script and everything else our app needs
# Expects a local copy of huggingface.co/EMBEDDIA/sloberta-tweetsentiment model in sloberta-tweetsentiment folder
COPY --chown=elg:elg sloberta-tweetsentiment /elg/sloberta-tweetsentiment/
COPY --chown=elg:elg docker-entrypoint.sh elg_service.py /elg/

ENV WORKERS=1
ENV TIMEOUT=30
ENV WORKER_CLASS=sync
ENV LOGURU_LEVEL=INFO


RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
