import numpy as np
import sympy as sp
import re

def print_sym_matrix(M, name='M'):
    print("\n")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Mij = sympy_to_numpy_str(str(M[i, j]))
            print(f"{name}[{i}][{j}] += {Mij}")


def sympy_to_numpy_str(expr):
    """
    将 SymPy 表达式字符串中的数学函数替换为 NumPy 版本
    返回可直接在 NumPy 中使用的字符串
    """
    # 核心替换字典（可扩展）
    func_replacements = {
        r'sin\(': 'np.sin(',
        r'cos\(': 'np.cos(',
        r'tan\(': 'np.tan(',
        r'asin\(': 'np.arcsin(',
        r'acos\(': 'np.arccos(',
        r'atan\(': 'np.arctan(',
        r'sinh\(': 'np.sinh(',
        r'cosh\(': 'np.cosh(',
        r'tanh\(': 'np.tanh(',
        r'asinh\(': 'np.arcsinh(',
        r'acosh\(': 'np.arccosh(',
        r'atanh\(': 'np.arctanh(',
        r'exp\(': 'np.exp(',
        r'log\(': 'np.log(',
        r'sqrt\(': 'np.sqrt(',
        r'pi': 'np.pi',
        r'E': 'np.e',
        r'atan2\(': 'np.arctan2(',
        r'Matrix': 'np.array',  # 替换矩阵表示
        r'Abs\(': 'np.abs('
    }
    
    # 处理函数简写情况（如 sin x → sin(x))
    expr = standardize_function_calls(expr)
    
    # 进行替换（按优先级从长到短排序）
    for pattern, replacement in sorted(func_replacements.items(), 
                                        key=lambda x: len(x[0]), 
                                        reverse=True):
        expr = re.sub(pattern, replacement, expr)
    
    # 特殊处理幂运算
    expr = re.sub(r'\*\*', '**', expr)
    
    return expr


def standardize_function_calls(expr_str):
    """
    标准化函数调用格式：确保所有函数调用都有括号
    如转换 "sin x" 为 "sin(x)"
    """
    # 识别常见的数学函数名
    math_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'asin', 'acos', 
                  'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'abs']
    
    # 按长度从长到短排序（优先匹配长函数名）
    func_pattern = '|'.join(sorted(math_funcs, key=len, reverse=True))
    
    # 查找后跟非(字符的函数名
    def add_parentheses(match):
        func_name = match.group(0)
        return f"{func_name}("
    
    # 处理简单函数调用（无参数）
    expr_str = re.sub(rf'\b({func_pattern})\b(?![\(])', add_parentheses, expr_str)
    
    return expr_str



if __name__ == '__main__':
    x1, x2 = sp.symbols('x1 x2')
    alpha = sp.symbols('alpha')

    x1_dot = x2
    sqrt_arg = 1 - (x2 / 15.06) ** 2
    x2_dot = 1.122954276 * sp.exp(-(x1 + 110) / 24) * sp.sqrt(sqrt_arg) * (alpha + 1.802274) - 0.02069 * sqrt_arg
    
    X = sp.Matrix([
        [x1],
        [x2],
    ])
    fxu = sp.Matrix([
        [x1_dot],
        [x2_dot],
    ])

    pfpx = fxu.jacobian(X)

    print_sym_matrix(pfpx, name='pfpx')

    print("ok")