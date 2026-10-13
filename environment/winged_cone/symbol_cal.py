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
    x1, x2 = sp.symbols('x1 x2')
    alpha = sp.symbols('alpha')

    x1_dot = x2
    sqrt_arg = 1 - (x2 / 15060.0) ** 2
    x2_dot = 64345.28 * sp.exp(-(x1 + 110000.0) / 24000.0) * sp.sqrt(sqrt_arg) * (alpha + 0.0314533) - 20.69 * sqrt_arg

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
