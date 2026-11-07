import numpy as np

def size(expr):
    """
    Returns the number of nodes in an expression tree.
    Numbers and variables count as 1.
    """
    if isinstance(expr, (int, float, str)):
        return 1
    if isinstance(expr, tuple):
        op = expr[0]
        if op == "neg":
            return 1 + size(expr[1])
        elif op in ("add", "mul"):
            return 1 + size(expr[1]) + size(expr[2])
    return 1


def _simplify_once(expr):
    """Applies single-step bottom-up algebraic rewrites."""
    if isinstance(expr, (int, float, str)):
        return expr
    
    op = expr[0]
    
    if op == "neg":
        a = _simplify_once(expr[1])
        if isinstance(a, (int, float)):
            return -a
        if isinstance(a, tuple) and a[0] == "neg":
            return a[1]
        return ("neg", a)
    
    elif op == "add":
        a = _simplify_once(expr[1])
        b = _simplify_once(expr[2])
        
        # Constant folding
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a + b
        
        # Identities: x + 0 -> x, 0 + x -> x
        if b == 0:
            return a
        if a == 0:
            return b
        
        # Canonical constants on the right: c + x -> x + c
        if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
            return ("add", b, a)
        
        # Chained constants: (x + c1) + c2 -> x + (c1 + c2)
        if isinstance(b, (int, float)) and isinstance(a, tuple) and a[0] == "add" and isinstance(a[2], (int, float)):
            x, c1 = a[1], a[2]
            return ("add", x, c1 + b)
        
        # Associativity toward the right: x + (y + c) -> (x + y) + c when y is not a constant
        if isinstance(b, tuple) and b[0] == "add" and isinstance(b[2], (int, float)):
            y, c = b[1], b[2]
            if not isinstance(y, (int, float)):
                return ("add", ("add", a, y), c)
                
        # Associativity toward the right: (x + c) + y -> (x + y) + c when y is not a constant
        if isinstance(a, tuple) and a[0] == "add" and isinstance(a[2], (int, float)):
            x, c = a[1], a[2]
            if not isinstance(b, (int, float)):
                return ("add", ("add", x, b), c)
        
        return ("add", a, b)
    
    elif op == "mul":
        a = _simplify_once(expr[1])
        b = _simplify_once(expr[2])
        
        # Constant folding
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a * b
        
        # Identities: x * 1 -> x, 1 * x -> x, x * 0 -> 0, 0 * x -> 0
        if b == 1:
            return a
        if a == 1:
            return b
        if b == 0 or a == 0:
            return 0
        
        # Canonical constants on the right: c * x -> x * c
        if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
            return ("mul", b, a)
        
        # Chained constants: (x * c1) * c2 -> x * (c1 * c2)
        if isinstance(b, (int, float)) and isinstance(a, tuple) and a[0] == "mul" and isinstance(a[2], (int, float)):
            x, c1 = a[1], a[2]
            return ("mul", x, c1 * b)
        
        # Distribution of a constant over a sum: (a + b) * c -> a * c + b * c
        if isinstance(a, tuple) and a[0] == "add" and isinstance(b, (int, float)):
            sum_left, sum_right = a[1], a[2]
            return ("add", ("mul", sum_left, b), ("mul", sum_right, b))
            
        if isinstance(b, tuple) and b[0] == "add" and isinstance(a, (int, float)):
            sum_left, sum_right = b[1], b[2]
            return ("add", ("mul", sum_left, a), ("mul", sum_right, a))
        
        return ("mul", a, b)
        
    return expr


def simplify(expr):
    """
    Simplifies an expression by repeatedly applying rewrite rules 
    bottom-up until a fixed point is reached.
    """
    prev = None
    curr = expr
    while prev != curr:
        prev = curr
        curr = _simplify_once(curr)
    return curr