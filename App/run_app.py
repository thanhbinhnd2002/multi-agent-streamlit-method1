# ✅ run_app.py
# Tập tin launcher dùng để mở ứng dụng Streamlit cho mô phỏng cạnh tranh ngoài

import os
import subprocess

# Tên file Streamlit chính (UI)
ENTRY_FILE = "App/UI.py"

# Lệnh để chạy ứng dụng streamlit
command = f"streamlit run {ENTRY_FILE}"

# Chạy lệnh trong terminal
subprocess.run(command, shell=True)
