import torch

def multiplication(x,y,p):
    return x*y
def addition(x,y,p):
    return x+y
def division(x,y,p):
    return (x * pow(y, p-2, p))
def difference(x,y,p):
    return x-y

composite={"xy": multiplication, "x+y": addition, "div": division}

def generate_data(p, operation):
    x = torch.arange(p)
    y = torch.arange(p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * p
    op = torch.ones_like(x) * (p+1)
    print(x.shape)
    if operation=="rand":
        result=torch.randint(0, p, size=(p*p,))
    elif operation==division:
        result=torch.empty_like(x)
        for i,row in enumerate(zip(x,y)):     
            result[i]=division(row[0].item(), row[1].item(), p)%p
    else:
        result = operation(x,y,p) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])
