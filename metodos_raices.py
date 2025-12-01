import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Callable, List, Tuple, Dict, Optional

def crear_funcion(expr_str: str, var: str = 'x') -> Tuple[Callable, Callable, sp.Expr]:
    """Convierte una expresión string en función numérica y simbólica"""
    x = sp.symbols(var)
    try:
        expr = sp.sympify(expr_str, locals={'e': sp.E, 'pi': sp.pi})
    except:
        raise ValueError("Ecuación no válida. Usa sintaxis Python/SymPy (ej: x**3 - 2*x - 5)")
    
    f_num = sp.lambdify(x, expr, 'numpy')
    f_prime = sp.diff(expr, x)
    f_prime_num = sp.lambdify(x, f_prime, 'numpy')
    
    return f_num, f_prime_num, expr

def biseccion(f: Callable, a: float, b: float, tol: float = 1e-10, max_iter: int = 100) -> Dict:
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    
    iteraciones = []
    error = tol + 1
    i = 0
    c_anterior = None
    
    while error > tol and i < max_iter:
        c = (a + b) / 2
        fc = f(c)
        
        if c_anterior is not None:
            error = abs(c - c_anterior)
        else:
            error = abs(b - a)
            
        iteraciones.append({
            'n': i+1,
            'a': a,
            'b': b,
            'c': c,
            'f(c)': fc,
            'error': error if i > 0 else None
        })
        
        if fc == 0:
            break
        elif f(a) * fc < 0:
            b = c
        else:
            a = c
            
        c_anterior = c
        i += 1
    
    return {
        'raiz': c,
        'iteraciones': iteraciones,
        'convergio': error <= tol,
        'metodo': 'Bisección'
    }

def newton_raphson(f: Callable, df: Callable, x0: float, tol: float = 1e-10, max_iter: int = 100) -> Dict:
    iteraciones = []
    x = x0
    error = tol + 1
    i = 0
    
    while error > tol and i < max_iter:
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivada cero en la iteración")
            
        x_new = x - fx / dfx
        error = abs(x_new - x)
        
        iteraciones.append({
            'n': i+1,
            'x': x,
            'f(x)': fx,
            'f\'(x)': dfx,
            'x_new': x_new,
            'error': error if i > 0 else None
        })
        
        x = x_new
        i += 1
    
    return {
        'raiz': x,
        'iteraciones': iteraciones,
        'convergio': error <= tol,
        'metodo': 'Newton-Raphson'
    }

def secante(f: Callable, x0: float, x1: float, tol: float = 1e-10, max_iter: int = 100) -> Dict:
    iteraciones = []
    x_prev = x0
    x = x1
    error = tol + 1
    i = 0
    
    while error > tol and i < max_iter:
        fx_prev = f(x_prev)
        fx = f(x)
        if fx - fx_prev == 0:
            raise ValueError("División por cero en secante")
            
        x_new = x - fx * (x - x_prev) / (fx - fx_prev)
        error = abs(x_new - x)
        
        iteraciones.append({
            'n': i+1,
            'x_{n-1}': x_prev,
            'x_n': x,
            'f(x_n)': fx,
            'x_{n+1}': x_new,
            'error': error if i > 0 else None
        })
        
        x_prev = x
        x = x_new
        i += 1
    
    return {
        'raiz': x,
        'iteraciones': iteraciones,
        'convergio': error <= tol,
        'metodo': 'Secante'
    }

def solucion_exacta(expr: sp.Expr, var='x'):
    x = sp.symbols(var)
    try:
        soluciones = sp.solve(expr, x)
        reales = [sol.evalf() for sol in soluciones if sol.is_real]  # Ya evalúa a float
        return reales[0] if reales else None
    except:
        return None