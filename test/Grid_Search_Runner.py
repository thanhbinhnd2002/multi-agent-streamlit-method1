import os
import subprocess
from itertools import product



# Các giá trị cần tìm kiếm dùng range # (có thể thay đổi theo nhu cầu)
# EPSILON_list = [0.05, 0.1, 0.2]
# DELTA_list   = [0.1, 0.2, 0.3 0.4]
# N_BETA_list  = [2, 3, 4]
EPSILON_list = range(0.05, 0.21, 0.05)  # Tạo dãy từ 0.05 đến 0.2 với bước 0.05
DELTA_list   = range(0.1, 0.41, 0.1)   # Tạo dãy từ 0.1 đến 0.4 với bước 0.1
N_BETA_list  = range(2, 5)              # Tạo dãy từ 2 đến 4 (bao gồm 2, 3, 4)

# Các tham số cố định
INF = 10000
MAX_ITER = 50
TOL = 1e-4

# Tên file mô phỏng chính (phải có sẵn)
SIMULATION_SCRIPT = "multi_Beta_Simulate_ver_2.py"

# Tạo tổ hợp và chạy
for EPSILON, DELTA, N_BETA in product(EPSILON_list, DELTA_list, N_BETA_list):
    folder_name = f"INF{INF}_EPS{EPSILON}_DELTA{DELTA}_ITER{MAX_ITER}_TOL{TOL}_NBETA{N_BETA}"
    output_folder = os.path.join("Output_test", folder_name)

    # Gọi script mô phỏng với biến môi trường
    env = os.environ.copy()
    env.update({
        "INF": str(INF),
        "EPSILON": str(EPSILON),
        "DELTA": str(DELTA),
        "MAX_ITER": str(MAX_ITER),
        "TOL": str(TOL),
        "N_BETA": str(N_BETA),
        "OUTPUT_FOLDER": output_folder
    })

    print(f"▶️ Chạy mô hình với: {folder_name}")
    subprocess.run(["python", SIMULATION_SCRIPT], env=env)
