FROM python:3.7-slim-buster
LABEL maintainer="v-crn"

RUN apt-get update && apt-get install -y python3-dev build-essential

ENV PYTHONUNBUFFERED 1
ENV WORKDIR /usr/src/south-park-dialogue
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR
RUN mkdir $WORKDIR/models

COPY Pipfile Pipfile.lock ./
COPY src/ ./src/
COPY data/ ./data/
RUN pip3 install pipenv --no-cache-dir && \
    pipenv install --system --deploy

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
