FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS build
COPY engine/requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED=1 GRADIO_SERVER_NAME=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/* \
 && useradd -m roveruser
COPY --from=build /deps /usr/local
WORKDIR /app
COPY --chown=roveruser engine/ .
USER roveruser
EXPOSE 7860
CMD ["python", "app.py"]
