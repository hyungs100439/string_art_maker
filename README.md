# String Art Maker

사진을 입력하면 핀과 실을 이용한 스트링 아트 도안을 자동으로 생성하는 파이썬 프로그램입니다.

## 필요한 패키지
```
pip install numpy pillow matplotlib scikit-image tqdm
```

## 사용 방법
1. `photo.jpg` 이미지를 `string_art.py`와 같은 폴더에 두세요.
2. 터미널에서 실행:
   ```
   python string_art.py
   ```
3. 실행이 끝나면 다음 파일이 생성됩니다:
   - `order.txt` : 핀 연결 순서
   - `result.png` : 결과 이미지
   - `result_with_pins.png` : 핀 번호가 표시된 이미지

## 주요 설정 (코드 하단에서 변경 가능)
- `num_nails`: 핀 개수
- `iterations`: 선 개수
- `darken_amount`: 한 줄의 영향도
- `gamma`: 결과 이미지 대비
