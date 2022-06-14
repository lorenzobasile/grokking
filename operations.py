import torch

def multiplication(x,y):
    return x*y
def addition(x,y):
    return x+y
def division(x,y):
    return x//y
def difference(x,y):
    return x-y

def x1(x,y):
    return x
def y1(x,y):
    return y
def x2(x,y):
    return x*x
def y2(x,y):
    return y*y
def x3(x,y):
    return x*x*x
def y3(x,y):
    return y*y*y

def x2y(x,y):
    return x*x*y
def xy2(x,y):
    return x*y*y

other={"x^2y":x2y, "xy^2": xy2}
monomial={"x":x1,"y":y1,"x^2": x2, "y^2": y2, "x^3": x3, "y^3": y3}
composite={"xy": multiplication, "x+y": addition, "xDIVy": division}

def generate_data(p, eq_token, op_token, operation):
    """
    x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = operation(x,y) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])
