import numpy as np
import sympy as sp
import re


def print_sym_matrix(M, name='M'):
    print("\n")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Mij = sympy_to_torch_str(str(M[i, j]))
            if Mij == '0':
                continue
            print(f"{name}[:, {i}, {j}] += {Mij}")


def sympy_to_numpy_str(expr):
    """
    Replace mathematical functions in a SymPy expression string with NumPy equivalents.
    Return a string that can be evaluated with NumPy.
    """
    # Core replacement map
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
        r'Matrix': 'np.array',
        r'Abs\(': 'np.abs('
    }
    
    # Handle abbreviated function notation
    expr = standardize_function_calls(expr)
    
    # Apply replacements
    for pattern, replacement in sorted(func_replacements.items(), 
                                        key=lambda x: len(x[0]), 
                                        reverse=True):
        expr = re.sub(pattern, replacement, expr)
    
    # Handle exponentiation
    expr = re.sub(r'\*\*', '**', expr)
    
    return expr


def sympy_to_torch_str(expr):
    """
    Replace mathematical functions in a SymPy expression string with NumPy equivalents.
    Return a string that can be evaluated with NumPy.
    """
    # Core replacement map
    func_replacements = {
        r'sin\(': 'torch.sin(',
        r'cos\(': 'torch.cos(',
        r'tan\(': 'torch.tan(',
        r'asin\(': 'torch.arcsin(',
        r'acos\(': 'torch.arccos(',
        r'atan\(': 'torch.arctan(',
        r'sinh\(': 'torch.sinh(',
        r'cosh\(': 'torch.cosh(',
        r'tanh\(': 'torch.tanh(',
        r'asinh\(': 'torch.arcsinh(',
        r'acosh\(': 'torch.arccosh(',
        r'atanh\(': 'torch.arctanh(',
        r'exp\(': 'torch.exp(',
        r'log\(': 'torch.log(',
        r'sqrt\(': 'torch.sqrt(',
        r'pi': 'torch.pi',
        r'E': 'torch.e',
        r'atan2\(': 'torch.arctan2(',
        r'Matrix': 'torch.array',
        r'Abs\(': 'torch.abs('
    }
    
    # Handle abbreviated function notation
    expr = standardize_function_calls(expr)
    
    # Apply replacements
    for pattern, replacement in sorted(func_replacements.items(), 
                                        key=lambda x: len(x[0]), 
                                        reverse=True):
        expr = re.sub(pattern, replacement, expr)
    
    # Handle exponentiation
    expr = re.sub(r'\*\*', '**', expr)
    
    return expr


def standardize_function_calls(expr_str):
    """
    Normalize function-call syntax so every call uses parentheses.
    For example, convert "sin x" to "sin(x)".
    """
    # Identify common mathematical functions
    math_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'asin', 'acos', 
                  'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'abs']
    
    # Sort by decreasing pattern length
    func_pattern = '|'.join(sorted(math_funcs, key=len, reverse=True))
    
    # Find function names not followed by an opening parenthesis
    def add_parentheses(match):
        func_name = match.group(0)
        return f"{func_name}("
    
    # Handle simple function calls
    expr_str = re.sub(rf'\b({func_pattern})\b(?![\(])', add_parentheses, expr_str)
    
    return expr_str



if __name__ == '__main__':
    x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = sp.symbols('x y z vx vy vz phi theta psi p q r')
    T, tau_x, tau_y, tau_z = sp.symbols('T tau_x tau_y tau_z')
    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9, lam10, lam11, lam12 = sp.symbols('lam1 lam2 lam3 lam4 lam5 lam6 lam7 lam8 lam9 lam10 lam11 lam12')
    m, g, Ixx, Iyy, Izz = sp.symbols('m g Ixx Iyy Izz')

    X = sp.Matrix([
        [x],
        [y],
        [z],
        [vx],
        [vy],
        [vz],
        [phi],
        [theta],
        [psi],
        [p],
        [q],
        [r]
    ])

    U = sp.Matrix([
        [T],
        [tau_x],
        [tau_y],
        [tau_z]
    ])

    sin_phi = sp.sin(phi)
    cos_phi = sp.cos(phi)
    sin_theta = sp.sin(theta)
    cos_theta = sp.cos(theta)
    tan_theta = sp.tan(theta)
    sin_psi = sp.sin(psi)
    cos_psi = sp.cos(psi)

    fxu = sp.zeros(12, 1)
    fxu[0, 0] += vx
    fxu[1, 0] += vy
    fxu[2, 0] += vz
    fxu[3, 0] += T * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) / m
    fxu[4, 0] += T * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) / m
    fxu[5, 0] += T * (cos_phi * cos_theta) / m - g
    fxu[6, 0] += p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    fxu[7, 0] += cos_phi * q - sin_phi * r
    fxu[8, 0] += sin_phi / cos_theta * q + cos_phi / cos_theta * r
    fxu[9, 0] += (1.0 / Ixx) * (tau_x + q * r * (Iyy - Izz))
    fxu[10, 0] += (1.0 / Iyy) * (tau_y + p * r * (Izz - Ixx))
    fxu[11, 0] += (1.0 / Izz) * (tau_z + p * q * (Ixx - Iyy))

    pfpx = fxu.jacobian(X)
    pfpu = fxu.jacobian(U)


    print_sym_matrix(pfpx, name='pfpx')
    print_sym_matrix(pfpu, name='pfpu')

    print("ok")
