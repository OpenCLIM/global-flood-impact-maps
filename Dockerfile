FROM continuumio/miniconda3:4.8.2
ENV PYTHONUNBUFFERED 1

RUN mkdir src

WORKDIR src

COPY environment.yml .
RUN conda env update -f environment.yml -n base

COPY singular_map.py .

CMD [ "python", "singular_map.py"]
