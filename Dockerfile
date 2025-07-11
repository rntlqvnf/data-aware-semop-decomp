# 1. 베이스 이미지 선택 (NVIDIA 공식, CUDA/PyTorch 사전 설치)
# 공식 레지스트리인 nvcr.io를 명시하여 '이미지 없음' 오류를 방지합니다.
FROM nvcr.io/nvidia/pytorch:24.03-py3

# 2. 컨테이너 내부의 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 먼저 복사 (도커 레이어 캐싱 최적화)
# requirements.txt가 변경되지 않으면 이 레이어는 재사용되어 빌드 속도가 빨라집니다.
COPY requirements.txt .

# 4. 시스템 라이브러리 및 파이썬 의존성 설치
# opencv-python 등에서 필요한 그래픽 관련 시스템 라이브러리를 먼저 설치합니다.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# 5. 프로젝트 소스 코드 전체 복사
# 의존성 설치 이후에 코드를 복사하여 캐싱 효율을 높입니다.
COPY . .