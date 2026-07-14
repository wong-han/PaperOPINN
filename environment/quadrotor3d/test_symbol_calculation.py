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
    phi, theta, psi, p, q, r = sp.symbols('phi theta psi p q r')
    lam1, lam2, lam3 = sp.symbols('lam1 lam2 lam3')
    q1, q2, q3, r1, r2, r3 = sp.symbols('q1 q2 q3 r1 r2 r3')
    Q = sp.diag(q1, q2, q3)
    R = sp.diag(r1, r2, r3)

    A = sp.zeros(3, 3)
    X = sp.Matrix([
        [phi],
        [theta],
        [psi]
    ])
    B = sp.Matrix([
        [1, sp.tan(theta)*sp.sin(phi), sp.tan(theta)*sp.cos(phi)],
        [0, sp.cos(phi), -sp.sin(phi)],
        [0, sp.sin(phi)/sp.cos(theta), sp.cos(phi)/sp.cos(theta)]
    ])
    U = sp.Matrix([
        [p],
        [q],
        [r]
    ])
    Lambda = sp.Matrix([
        [lam1],
        [lam2],
        [lam3]
    ])

    x_dot = A*X + B*U

    H = X.T * Q * X + U.T * R * U + Lambda.T * x_dot

    Lambda_dot = (-H.jacobian(X)).T

    Z = sp.Matrix([X, Lambda])
    Z_dot = sp.Matrix([x_dot, Lambda_dot])

    pFpZ = Z_dot.jacobian(Z)
    pFpU = Z_dot.jacobian(U)

    U_star = -.5 * R.inv() * B.T * Lambda
    pUpZ = U_star.jacobian(Z)


    print_sym_matrix(pFpZ, name='pFpZ')
    print_sym_matrix(pFpU, name='pFpU')
    print_sym_matrix(pUpZ, name='pUpZ')

    print("ok")