import numpy as np
from PIL import Image
import math
from skimage.draw import line as sk_line
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# 1. 이미지 로드 (0=흰색, 1=검정)
# -------------------------
def load_gray_norm(path, size=600):
    img = Image.open(path).convert('L').resize((size, size))
    arr = np.array(img).astype(np.float32)
    arr = 1.0 - arr / 255.0  # 흰=0, 검=1
    return arr

# -------------------------
# 2. 원형으로 못 좌표 생성
# -------------------------
def nails_on_circle(num_nails, radius, center):
    angles = np.linspace(0, 2*math.pi, num_nails, endpoint=False)
    return np.stack([
        center[0]+radius*np.cos(angles),
        center[1]+radius*np.sin(angles)
    ], axis=1).astype(int)

# -------------------------
# 3. 모든 nail 쌍에 대한 line 좌표 사전 계산
# -------------------------
def precompute_line_masks(image_shape, nails):
    masks = {}
    H,W = image_shape
    N = len(nails)
    for i in range(N):
        for j in range(i+1, N):
            r0, c0 = nails[i][1], nails[i][0]
            r1, c1 = nails[j][1], nails[j][0]
            rr, cc = sk_line(r0, c0, r1, c1)
            valid = (rr>=0)&(rr<H)&(cc>=0)&(cc<W)
            masks[(i,j)] = (rr[valid], cc[valid])
    return masks

# -------------------------
# 4. 메인 알고리즘
# -------------------------
def make_string_art(target, nails, masks,
                    iterations=2000, darken_amount=0.015,
                    candidate_limit=None):
    H,W = target.shape
    current = np.zeros_like(target)
    pos = 0
    order = []
    N = len(nails)

    for _ in tqdm(range(iterations)):
        best_score = -1
        best_idx = None
        rr_best, cc_best = None, None

        candidates = range(N)
        if candidate_limit:
            candidates = np.random.choice(N, size=min(candidate_limit,N), replace=False)

        for j in candidates:
            if j == pos: continue
            i0,i1 = sorted((pos,j))
            rr, cc = masks[(i0,i1)]
            vals = target[rr,cc] - current[rr,cc]
            score = np.sum(vals)
            if score > best_score:
                best_score = score
                best_idx = j
                rr_best, cc_best = rr, cc

        if best_idx is None: break
        current[rr_best, cc_best] = np.clip(current[rr_best, cc_best] + darken_amount, 0, 1)
        order.append((pos, best_idx))
        pos = best_idx

    return order, current

# -------------------------
# 5. 실행
# -------------------------
if __name__ == "__main__":
    # 입력 이미지와 설정
    target = load_gray_norm('photo.jpg', size=800)

    center = (400, 400)  # 중심 좌표
    nails = nails_on_circle(num_nails=300, radius=390, center=center)

    print("핀 쌍 선 마스크 계산 중 (최초 1회만 오래 걸립니다)...")
    masks = precompute_line_masks(target.shape, nails)

    # 알고리즘 실행
    order, current = make_string_art(
        target, nails, masks,
        iterations=10000,
        darken_amount=0.09,
        candidate_limit=None
    )

    print(f"총 {len(order)}개의 선 생성 완료!")

    # -------------------------
    # 연결 순서 파일 저장
    # -------------------------
    with open('order.txt', 'w') as f:
        for idx, (start, end) in enumerate(order, 1):
            f.write(f"{idx}: {start} -> {end}\n")
    print("order.txt 파일로 연결 순서를 저장했습니다.")

    # -------------------------
    # 결과 이미지 저장 (핀 번호 포함)
    # -------------------------
    gamma = 0.8
    result_image = (1-current)

    plt.figure(figsize=(8,8))
    plt.imshow(result_image, cmap='gray')
    plt.axis('off')

    # 핀 번호 표시 (원 바깥쪽으로 밀어서 표시)
    cx, cy = center
    factor = 1.08  # 1보다 크면 바깥쪽
    for i, (x, y) in enumerate(nails):
        dx = x - cx
        dy = y - cy
        tx = cx + dx * factor
        ty = cy + dy * factor
        plt.text(tx, ty, str(i),
                 color='red', fontsize=6,
                 ha='center', va='center')

    plt.savefig('result_with_pins.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    # 번호 없는 결과도 저장
    plt.imsave('result.png', (1-current)**gamma, cmap='gray', dpi=300)

    print("result_with_pins.png (핀 번호 포함) / result.png (결과 이미지) 저장 완료")
