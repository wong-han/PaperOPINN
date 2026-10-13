import re
import os

def process_unconstrained():
    file_path = "p4_quadrotor3d_TR_with_SIREN.py"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start_sim = -1
    end_sim = -1
    start_save = -1
    end_save = -1
    
    for i, line in enumerate(lines):
        if 'print("Starting Monte Carlo simulation...")' in line:
            start_sim = i
        elif 'df_box = pd.DataFrame({"Return": data_returns, "Controller": data_controllers})' in line:
            end_sim = i
        elif '# =============== 自动保存绘图数据 ===============' in line:
            start_save = i
        elif 'print(f"All Monte Carlo simulation data successfully saved to {data_dir}/")' in line:
            end_save = i
            
    if start_sim != -1 and end_sim != -1:
        for i in range(start_sim, end_sim + 1):
            lines[i] = "# " + lines[i]
            
    if start_save != -1 and end_save != -1:
        for i in range(start_save, end_save + 1):
            lines[i] = "# " + lines[i]
            
    # Insert loading code before fig, ax = plt.subplots...
    for i, line in enumerate(lines):
        if 'fig, ax = plt.subplots(figsize=(8, 7))' in line and not line.strip().startswith('#'):
            indent = line[:len(line) - len(line.lstrip())]
            load_code = f"{indent}df_box = pd.read_csv(os.path.join(data_dir, args.filename_prefix + \"return_boxplot_data.csv\"))\n"
            lines.insert(i, load_code)
            break
            
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
def process_constrained():
    file_path = "p4_quadrotor3d_TR_constrained_with_SIREN.py"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start_sim = -1
    end_sim = -1
    start_save = -1
    end_save = -1
    start_ret = -1
    end_ret = -1
    
    for i, line in enumerate(lines):
        if 'print("Starting Monte Carlo simulation...")' in line:
            start_sim = i
        elif '"Controller": data_controllers_logy' in line:
            end_sim = i + 2 # include closing brace
        elif '# =============== 自动保存绘图数据 ===============' in line:
            start_save = i
        elif 'print(f"All Monte Carlo simulation data successfully saved to {data_dir}/")' in line:
            end_save = i
        elif '# 画 Return 箱线图' in line:
            start_ret = i
        elif 'df_ret.to_csv' in line:
            end_ret = i
            
    if start_sim != -1 and end_sim != -1:
        for i in range(start_sim, min(end_sim + 1, len(lines))):
            lines[i] = "# " + lines[i]
            
    if start_save != -1 and end_save != -1:
        for i in range(start_save, end_save + 1):
            lines[i] = "# " + lines[i]
            
    if start_ret != -1 and end_ret != -1:
        for i in range(start_ret, end_ret + 1):
            lines[i] = "# " + lines[i]
            
    # Insert loading code for logy
    for i, line in enumerate(lines):
        if 'fig_y, ax_y = plt.subplots(figsize=(8, 7))' in line and not line.strip().startswith('#'):
            indent = line[:len(line) - len(line.lstrip())]
            load_code = f"{indent}df_logy = pd.read_csv(os.path.join(data_dir, args.filename_prefix + \"logy_boxplot_data.csv\"))\n"
            lines.insert(i, load_code)
            break
            
    # Insert loading code for ret
    for i, line in enumerate(lines):
        if 'fig_r, ax_r = plt.subplots(figsize=(8, 7))' in line and not line.strip().startswith('#'):
            indent = line[:len(line) - len(line.lstrip())]
            load_code = f"{indent}df_ret = pd.read_csv(os.path.join(data_dir, args.filename_prefix + \"return_boxplot_data.csv\"))\n"
            lines.insert(i, load_code)
            break
            
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
process_unconstrained()
process_constrained()
print("Done")
