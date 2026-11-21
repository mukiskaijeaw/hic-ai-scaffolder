# Hugging Face Space (SDK: Docker). Builds hicscaf Studio from the GitHub repo.
# Set the password in the Space: Settings > Variables and secrets > New secret > HICSCAF_PASSWORD
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
ENV HOME=/home/user PYTHONUNBUFFERED=1 \
    HICSCAF_PUBLIC=1 HICSCAF_HOST=0.0.0.0 PORT=7860 \
    HICSCAF_MAX_UPLOAD_MB=4096 HICSCAF_MAX_JOBS=1
WORKDIR /app
# change the URL if the repository moves; "Factory rebuild" in the Space pulls the latest main
RUN git clone --depth 1 https://github.com/mukiskaijeaw/hic-ai-scaffolder /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e .
USER user
EXPOSE 7860
CMD ["python", "-m", "hicscaf.cli", "studio", "--no-browser", "--workdir", "/home/user/hicscaf_studio"]
