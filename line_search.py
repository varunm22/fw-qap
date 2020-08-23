import numpy as np
from utils import *

def line_search(stype, x, d, g, A, B):
    # TODO: think about these other cases and maybe implement them?
    if stype == 0: # shortcut where we just set salpha to 1? not sure
        pass
    elif stype == 1: # bisection search, not assuming quadratic?
        pass
    elif stype == 2: # assume quadratic, search by derivative
        b = g.dot(d)
        c = fun(x,A,B)
        fun_vertex = fun(x+d,A,B)
        a = fun_vertex - b - c
        if (abs(a) < np.spacing(1)): # function is linear
            salpha = 1.
        else:
            salpha = min(1,max(-b/(2*a),0))
        fun_alpha = fun(x+salpha*d,A,B)

        # this code to "check quadratic function"?
        qfun_alpha = a*salpha*salpha + b*salpha + c
        if ((abs(a) >= np.spacing(1))
            and (abs(fun_alpha-qfun_alpha)>1000*abs(fun_alpha)*np.spacing(1))):
            raise "quadratic search error"

        # TODO: understand this side case
        if (fun_alpha>c):
            salpha = 0
            fun_alpha = c

        # TODO: understand this side case
        if (fun_alpha > fun_vertex):
            salpha = 1
            fun_alpha = fun_vertex
            f0new = fun_alpha

    else:
        raise "unsupported stype for line search"

    xt = x + salpha*d
    f0new = fun(xt, A, B)

    return (f0new, salpha)


