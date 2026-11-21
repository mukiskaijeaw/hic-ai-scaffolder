# hicscaf Studio as a hosted web app (Hugging Face Spaces, Render, Fly.io, any VM with Docker).
#   docker build -t hicscaf-studio .
#   docker run -p 7860:7860 -e HICSCAF_PASSWORD=choose-one hicscaf-studio
# then open http://localhost:7860 (user name: anything, password: the one you chose)
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user PYTHONUNBUFFERED=1 \
    HICSCAF_PUBLIC=1 HICSCAF_HOST=0.0.0.0 PORT=7860 \
    HICSCAF_MAX_UPLOAD_MB=4096 HICSCAF_MAX_JOBS=1
WORKDIR /app
COPY . /app
# CPU-only PyTorch keeps the image ~1 GB instead of ~5 GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e .
USER user
EXPOSE 7860
CMD ["python", "-m", "hicscaf.cli", "studio", "--no-browser", "--workdir", "/home/user/hicscaf_studio"]
