import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# --- 모델 파라미터 및 상품 구조 정의 ---
product_spec = {
    "expiry": 1.0,
    "strikes": [0.0040, 0.0080, 0.0120],   # 40bp, 80bp, 120bp
    "step_coupons": [0.03, 0.03, 0.02],    # 각 레벨 통과 시 추가되는 쿠폰
}
VOL = 0.00868544343838943 ** 2  # 연간 변동성 (100bp)
DISCOUNT_FACTOR = 0.9641638309051358 # 할인계수 P(0, T_1)

# --- 첨부된 이론에 따른 디지털 쿠폰 본드 그릭스 계산 함수 ---
# (Theorem 5. Digital Coupon Bond Greeks Formulas)
def get_digital_bond_greeks(S, K, T, sigma, P):
    """
    단일 디지털 쿠폰 본드의 그릭스를 계산합니다.
    """
    if T <= 1e-9: # 만기 시점 처리
        delta = gamma = vega = theta = 0
        return delta, gamma, vega, theta

    sqrt_T = np.sqrt(T)
    d = (S - K) / (sigma * sqrt_T)
    
    phi_d = st.norm.pdf(d)
    
    # Theorem 5의 공식 적용
    delta = P * (phi_d / (sigma * sqrt_T))
    gamma = P * (-d * phi_d / (sigma**2 * T))
    vega = P * (-d * phi_d / sigma) * sqrt_T # 원본 이론의 베가 공식 수정 (∂d/∂σ)
    theta = P * (d * phi_d / (2 * T * sigma * sqrt_T))

    return delta, gamma, vega, theta

# --- 스텝업 구조(포트폴리오)의 전체 그릭스 계산 함수 ---
def get_step_up_bond_greeks(S, spec, T, sigma, P):
    """
    여러 디지털 본드로 구성된 스텝업 본드의 전체 그릭스를 계산합니다.
    """
    total_delta, total_gamma, total_vega, total_theta = 0, 0, 0, 0
    
    for i in range(len(spec["strikes"])):
        K = spec["strikes"][i]
        coupon = spec["step_coupons"][i]
        
        # 각 디지털 본드의 그릭스를 계산
        delta, gamma, vega, theta = get_digital_bond_greeks(S, K, T, sigma, P)
        
        # 쿠폰 크기를 가중치로 하여 합산
        total_delta += coupon * delta
        total_gamma += coupon * gamma
        total_vega += coupon * vega
        total_theta += coupon * theta
        
    return total_delta, total_gamma, total_vega, total_theta

# --- 시각화를 위한 데이터 생성 ---
spreads = np.linspace(0.0, 0.02, 400) # 0bp to 200bp
deltas, gammas, vegas, thetas = [], [], [], []

for s in spreads:
    d, g, v, t = get_step_up_bond_greeks(
        s, product_spec, product_spec["expiry"], VOL, DISCOUNT_FACTOR
    )
    deltas.append(d)
    gammas.append(g)
    vegas.append(v)
    thetas.append(t)

# --- 2x2 플롯 생성 ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Digital Step-up Bond Greeks (Bachelier Model)', fontsize=16)
spreads_in_bp = spreads * 10000 # bp 단위로 변환

# 1. Delta Plot
axs[0, 0].plot(spreads_in_bp, deltas, color='blue')
axs[0, 0].set_title('Delta')
axs[0, 0].set_ylabel('Delta')
axs[0, 0].grid(True, linestyle='--', alpha=0.6)

# 2. Gamma Plot
axs[0, 1].plot(spreads_in_bp, gammas, color='green')
axs[0, 1].set_title('Gamma')
axs[0, 1].set_ylabel('Gamma')
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

# 3. Vega Plot
axs[1, 0].plot(spreads_in_bp, vegas, color='red')
axs[1, 0].set_title('Vega')
axs[1, 0].set_xlabel('Underlying Spread (bp)')
axs[1, 0].set_ylabel('Vega')
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

# 4. Theta Plot
axs[1, 1].plot(spreads_in_bp, thetas, color='purple')
axs[1, 1].set_title('Theta')
axs[1, 1].set_xlabel('Underlying Spread (bp)')
axs[1, 1].set_ylabel('Theta')
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

# 행사가 수직선 추가
for ax in axs.flat:
    for strike in product_spec["strikes"]:
        ax.axvline(x=strike*10000, color='gray', linestyle=':', linewidth=1.2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()