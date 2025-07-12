# 1. 베이스 이미지 선택 (NVIDIA 공식, CUDA/PyTorch 사전 설치)
FROM nvcr.io/nvidia/pytorch:24.03-py3

# 2. 컨테이너 내부의 작업 디렉토리 설정
WORKDIR /app

# 3. 💡 Azure CLI 설치 (추가된 부분)
# Microsoft의 공식 설치 스크립트를 사용하여 컨테이너 내부에 Azure CLI를 설치합니다.
# 이렇게 하면 컨테이너 안에서 'az' 명령어를 사용할 수 있게 됩니다.
RUN apt-get update && apt-get install -y curl \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사
COPY requirements.txt .

# 5. 시스템 라이브러리 및 파이썬 의존성 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# 6. 프로젝트 소스 코드 전체 복사
COPY . .
