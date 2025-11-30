class Elemento:
    def __init__(self, n, p):
        if p <= 1:
            raise ValueError("p debe ser un número primo mayor que 1.")
        self.representante = n % p
        self.valor = n
        self.p = p

    def mismo_Fp(self, other):
        if self.p != other.p:
            raise ValueError("Los elementos no pertenecen al mismo cuerpo Fp.")

    def __eq__(self, other):
        if not isinstance(other, Elemento):
            return NotImplemented
        self.mismo_Fp(other)
        return self.representante == other.representante

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __neg__(self):
        return Elemento(self.p-self.representante, self.p)

    def __add__(self, other):
        self.mismo_Fp(other)
        return Elemento(self.representante + other.representante, self.p)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        self.mismo_Fp(other)
        return Elemento(self.representante * other.representante, self.p)

    def __pow__(self, k):
        return Elemento(pow(self.representante, k, self.p), self.p)

    def __str__(self):
        return f"{self.representante} mod {self.p}"

    def __repr__(self):
        return str(self)

    def __int__(self):
        return self.representante

import random

class cuerpo_fp:
    
    def __init__(self, p):
        self.p = p
   
    def de_Fp(self, a, b):
        if self.p != a.p or self.p != b.p:
            raise Exception(f'Los elementos {a} y {b} no pertenecen al mismo cuerpo.')
   
    def cero(self):
        return Elemento(0, self.p)
   
    def uno(self):
        return Elemento(1, self.p)
   
    def elem_de_int(self, n):
        return Elemento(n, self.p)
   
    def elem_de_str(self, s):
        return self.elem_de_int(int(s))
   
    def conv_a_int(self, a):
        return int(a)
   
    def conv_a_str(self, a):
        return str(a)
   
    def suma(self, a, b):
        return a.__add__(b)
   
    def inv_adit(self, a):
        return a.__neg__()
    
    def mult(self, a, b):     # a*b
        return a.__mul__(b)
    
    def pot(self, a, k):      # a^k (k entero)
        return a.__pow__(k)
    
    def inv_mult(self, a):    # a^(-1)
        if self.es_cero(a):
            raise ZeroDivisionError(f"No se puede calcular el inverso multiplicativo de 0 en F{self.p}.")
        return a.__pow__(self.p-2)

    def es_cero(self, a):
        self.de_Fp(a, self.cero())
        return a.__eq__(self.cero())
   
    def es_uno(self, a):
        self.de_Fp(a, self.uno())
        return a.__eq__(self.uno())
   
    def es_igual(self, a, b):
        self.de_Fp(a, b)
        return a.__eq__(b)
   
    def aleatorio(self):
        return self.elem_de_int(random.randint(0, self.p - 1))

    def tabla_suma(self):
            print(f"Tabla de suma en F{self.p}:")
            print("     ", end="")
            for j in range(self.p):
                print(f"{j:4}", end="")
            print("\n" + "-" * (6 + 4 * self.p))
            for i in range(self.p):
                print(f"{i:2} |", end="")
                for j in range(self.p):
                    a = self.elem_de_int(i)
                    b = self.elem_de_int(j)
                    resultado = self.suma(a, b)
                    print(f"{int(resultado):4}", end="")
                print()

    def tabla_multiplicacion(self):
        print(f"Tabla de multiplicación en F{self.p}:")
        print("     ", end="")
        for j in range(self.p):
            print(f"{j:4}", end="")
        print("\n" + "-" * (6 + 4 * self.p))
        for i in range(self.p):
            print(f"{i:2} |", end="")
            for j in range(self.p):
                a = self.elem_de_int(i)
                b = self.elem_de_int(j)
                resultado = self.mult(a, b)
                print(f"{int(resultado):4}", end="")
            print()

    def tabla_inv_adit(self):
            inversos = []
            print(f"Inversos aditivos en F{self.p}:")
            for i in range(self.p):
                a = self.elem_de_int(i)
                inv = self.inv_adit(a)
                inversos.append(inv)
                print(f"{i} -> {int(inv)}")
            return inversos

    def tabla_inv_mult(self):
            inversos = ["*"]
            print(f"Inversos multiplicativos en F{self.p}:")
            print("0 -> * (no tiene inverso)")
   
            for i in range(1, self.p):
                a = self.elem_de_int(i)
                try:
                    inv = self.inv_mult(a)
                    inversos.append(inv)
                    print(f"{i} -> {int(inv)}")
                except ZeroDivisionError:
                    inversos.append(None) 
                    print(f"{i} -> ERROR (No tiene inverso)")
            return inversos
           
    def cuadrado_latino(self, a):
        if self.es_cero(a):
            raise ValueError("a no puede ser 0 para formar un cuadrado latino.")
        print(f"Cuadrado latino generado con a = {int(a)} en F{self.p}:\n")
        cuadrado = []
        print("     ", end="")
        for j in range(self.p):
            print(f"{j:4}", end="")
        print("\n" + "-" * (6 + 4 * self.p))
        for i in range(self.p):
            fila = []
            print(f"{i:2} |", end="")
            for j in range(self.p):
                valor = self.suma(self.mult(a, self.elem_de_int(i)), self.elem_de_int(j))
                fila.append(valor)
                print(f"{self.conv_a_int(valor):4}", end="")
            cuadrado.append(fila)
            print()
        return cuadrado

import re

class Polinomio: #Daremos por supuesto que los coeficientes son de la clase Elemento 
    def __init__(self, fp, Cn, var='x'):
        self.p = fp.p
        self.fp = fp
        self.coef = self._normalizar(Cn)
        self.var = var
        self.deg = len(self.coef)-1
        
    def _normalizar(self, Cn): # elimina 0s de los coeficientes nulos y suponemos que la entrada es una lista
        c = Cn[:]
        while len(c) > 1 and self.fp.es_cero(c[-1]):
            c.pop()
        if not c:
            c = [self.fp.cero()]
        return c
    
    @classmethod
    def desde_tuple(cls, fp, t, var = 'x'):
        if not t: 
             return cls(fp, [fp.cero()], var)
        elif isinstance(t[0], Elemento):
            coefs = [c for c in t]
            return cls(fp, coefs, var)
        elif isinstance(t[0], int):
            coefs = [fp.elem_de_int(c) for c in t]
            return cls(fp, coefs, var)
        else:
            raise NotImplementedError
 
    @classmethod
    def desde_entero(cls, fp, n, var='x'):
        coefs = []
        while n > 0:
            coefs.append(fp.elem_de_int(n % fp.p))
            n = n // fp.p
        if not coefs:
            coefs = [fp.cero()]
        return cls(fp, coefs, var)

    @classmethod
    def desde_str(cls, fp, s, var='x'):
        s = s.replace(" ", "")
        if not s: 
            return cls(fp, [fp.cero()], var)
        if s[0] not in "+-":
            s = "+" + s
        terminos = re.findall(r'[+-][^+-]+', s)
        coef_dict = {}
        for t in terminos:
            signo = -1 if t[0] == '-' else 1
            t = t[1:]
            if var not in t:
                coef = int(t) if t else 1
                exp = 0
            else:
                partes = t.split(var)
                if partes[0] in ('', '+', '*'):
                    coef = 1
                elif partes[0] == '-':
                    coef = -1
                elif partes[0].endswith('*'):
                    coef = int(partes[0][:-1])
                else:
                    coef = int(partes[0])
                if len(partes) == 1 or partes[1] == '':
                    exp = 1
                elif partes[1].startswith('^'):
                    exp = int(partes[1][1:])
                elif partes[1].startswith('**'):
                    exp = int(partes[1][2:])
                else:
                    exp = 1
            coef_dict[exp] = coef_dict.get(exp, 0) + signo * coef

        grado_max = max(coef_dict.keys()) if coef_dict else 0
        coefs = [fp.elem_de_int(coef_dict.get(i, 0)) for i in range(grado_max + 1)]
        return cls(fp, coefs, var)

    def __str__(self):
        s = []
        for i, a in enumerate(reversed(self.coef)):
            exp = self.deg - i
            if self.fp.es_cero(a):
                continue          
            term = ""
            coef_val = self.fp.conv_a_int(a)
            if coef_val > self.p // 2: 
                 coef_val = coef_val - self.p
            signo = "+" if coef_val > 0 else "-"
            coef_abs = abs(coef_val)
            if exp == self.deg: 
                signo = "-" if coef_val < 0 else ""
            if coef_abs == 1 and exp != 0:
                coef_str = ""
            else:
                coef_str = str(coef_abs)
            if exp == 0:
                var_str = ""
            elif exp == 1:
                var_str = self.var
            else:
                var_str = f"{self.var}^{exp}"
            
            term = f" {signo} {coef_str}{var_str}"

            s.append(term)
            
        if not s:
            return "0"
        
        result = "".join(s).strip()
        if result.startswith("+"):
            result = result[1:].strip()
        return result
        
    def __iter__(self):
        return iter(self.coef)

    def __getitem__(self, i):
        return self.coef[i]

    def __int__(self):
        resultado = 0
        for i in range(len(self.coef)-1,-1,-1):
            resultado = resultado*self.p+int(self.coef[i])
        return resultado

    def __repr__(self):
        return str(self)

    def __add__(self, other):

        if self.p != other.p:
            raise ValueError("Los polinomios deben pertenecer al mismo cuerpo Fp")
        elif self.var != other.var:
            raise ValueError('Los polinomios han de tener la misma variable muda.')
        grado_max = max(self.deg, other.deg)
        suma_coef = [self.fp.cero() for _ in range(grado_max + 1)]
        for i in range(self.deg+1):
            suma_coef[i] = suma_coef[i] + self.coef[i]
        for i in range(other.deg+1):
            suma_coef[i] = suma_coef[i] + other.coef[i]
        return Polinomio(self.fp, suma_coef, self.var)
    
    def __neg__(self):
        inverso = [-c for c in self.coef]
        return Polinomio(self.fp, inverso, self.var)
    
    def __sub__(self, other):
        return self.__add__(other.__neg__())
    
    #     return Polinomio(Fp, nuevo_coef, self.var)

    def __mul__(self, other):
        
        if self.p != other.p:
            raise ValueError("Los polinomios deben pertenecer al mismo cuerpo Fp")
        elif self.var != other.var:
            raise ValueError('Los polinomios han de tener la misma variable muda.')
        m = self.deg
        n = other.deg
        nuevo_coef = [self.fp.cero() for _ in range(m + n + 1)]
        for i in range(m+1):
            for j in range(n+1):
                nuevo_coef[i + j] = nuevo_coef[i + j] + (self.coef[i] * other.coef[j])
    
        return Polinomio(self.fp, nuevo_coef, self.var)
        
    def divmod(self, other):

        if self.p != other.p:
            raise ValueError("Los polinomios deben pertenecer al mismo cuerpo Fp")
        elif self.var != other.var:
            raise ValueError('Los polinomios han de tener la misma variable muda.')
            
        A = self.coef[:]  
        B = other.coef[:]
    
        while len(A) > 1 and self.fp.es_cero(A[-1]):
            A.pop()
        while len(B) > 1 and self.fp.es_cero(B[-1]):
            B.pop()

        if len(B) == 0 or all(self.fp.es_cero(c) for c in B):
            raise ZeroDivisionError("División por polinomio nulo")
    
        degA = len(A) - 1
        degB = len(B) - 1

        if degA < degB:
            return (
                Polinomio(self.fp, [self.fp.cero()], self.var), 
                Polinomio(self.fp, A, self.var)
            )
    
        Q = [self.fp.cero() for _ in range(degA - degB + 1)]
        R = A[:]     
        inv_lider_B = self.fp.inv_mult(B[-1])  

        for k in range(degA - degB, -1, -1):
            coeff = R[degB + k] * inv_lider_B
            Q[k] = coeff
    
            for j in range(degB + 1):
                R[j + k] = R[j + k] - coeff * B[j]
    
        while len(R) > 1 and self.fp.es_cero(R[-1]):
            R.pop()
    
        return Polinomio(self.fp, Q, self.var), Polinomio(self.fp, R, self.var)

    def __floordiv__(self, other):

        Q, _ = self.divmod(other)
        return Q
    
    def __mod__(self, other):

        _, R = self.divmod(other)
        return R

    def __eq__(self, other):
        if isinstance(other, Polinomio):
            if self.fp.p != other.fp.p or self.var != other.var:
                return False
            return self.coef == other.coef
        
        if isinstance(other, int):
            if other == 0: return self.es_cero()
            elif other == 1: return self.es_uno()
            else: return self.__eq__(Polinomio.desde_entero(self.fp, other, self.var))
    
        return NotImplemented
    
    def es_cero(self):
        return self.deg == 0 and self.fp.es_cero(self.coef[0])
    
    def es_uno(self):
        return self.deg == 0 and self.fp.es_uno(self.coef[0])
    
    def coef_lider(self):
        return self.coef[-1]
    
    def factores(self):
    
        d = self.deg      
        if d <= 1:
            return []  
        factors = []

        if d % 2 == 0:
            factors.append(2)
            while d % 2 == 0:
                d //= 2 

        p = 3
        while p * p <= d:
            if d % p == 0:
                factors.append(p)
                while d % p == 0:
                    d //= p
            p += 2 
            
        if d > 1:
            factors.append(d)
            
        return factors
    
    def derivada(self):
        if self.deg == 0:
            return Polinomio(self.fp, [self.fp.cero()], self.var)

        nuevos_coef = [self.fp.cero()] * self.deg 

        for i in range(1, self.deg + 1):

            nuevos_coef[i-1] = self.coef[i] * self.fp.elem_de_int(i)
            
        return Polinomio(self.fp, nuevos_coef, self.var)

class anillo_fp_x:
    
    def __init__(self, fp, var='x'): # construye el anillo Fp[var]
        self.fp = fp
        self.var = var
        
    def cero(self):                 # 0
        return Polinomio(self.fp, [self.fp.cero()], self.var)
    
    def uno(self):                  # 1
        return Polinomio(self.fp, [self.fp.uno()], self.var)
    
    def elem_de_tuple(self, a):     # fabrica un polinomio a partir de la tupla (a0, a1, ...)
        return Polinomio.desde_tuple(self.fp, a, self.var)
    
    def elem_de_int(self, a):       # fabrica un polinomio a partir de los dígitos de a en base p
        return Polinomio.desde_entero(self.fp, a, self.var)
    
    def elem_de_str(self, s):       # fabrica un polinomio a partir de un string (parser)
        return Polinomio.desde_str(self.fp, s, self.var)
    
    def conv_a_tuple(self, a):      # devuelve la tupla de coeficientes
        return tuple(a)

    def conv_a_int(self, a):
        return int(a)
    
    def conv_a_str(self, a):        # pretty-printer
        return str(a)
    
    def suma(self, a, b):           # a+b
        return a.__add__(b)
    
    def inv_adit(self, a):          # -a
        return a.__neg__()
    
    def mult(self, a, b):           # a*b
        if a.deg == 0:
            return self.mult_por_escalar(b, int(a))
        elif b.deg == 0:
            return self.mult_por_escalar(a, int(b))
        return a.__mul__(b)
    
    def mult_por_escalar(self, a, e): # a*e (con e en Z/pZ)
        if not isinstance(e, Elemento): e = self.fp.elem_de_int(e)
        Cn = [e*c for c in a]
        return Polinomio(self.fp, Cn, self.var)
    
    def divmod(self, a, b):        # devuelve q,r tales que a=bq+r y deg(r)<deg(b)
        return a.divmod(b)
    
    def div(self, a, b):           # q
        return a.__floordiv__(b)
    
    def mod(self, a, b):           # r
        return a.__mod__(b)
    
    def grado(self, a):            # deg(a)
        return a.deg
    
    def gcd(self, a, b):           # devuelve g = gcd(a,b) mónico
        while not self.es_cero(b):
            a, b = b, self.mod(a, b)

        if self.es_cero(a):
            return self.cero()
        lider = a.coef_lider()
        inverso_lider = self.fp.inv_mult(lider)
        return self.mult_por_escalar(a, inverso_lider)
        
    def gcd_ext(self, a, b):       # devuelve g,x,y tales que g=ax+by, g=gcd(a,b) mónico
        cero = self.cero()
        uno = self.uno()
        r_prev, r_curr = a, b
        s_prev, s_curr = uno, cero
        t_prev, t_curr = cero, uno

        while not self.es_cero(r_curr):
            q, r_nuevo = self.divmod(r_prev, r_curr)
            r_prev, r_curr = r_curr, r_nuevo 
            s_prev, s_curr = s_curr, self.suma(s_prev, self.inv_adit(self.mult(q, s_curr)))
            t_prev, t_curr = t_curr, self.suma(t_prev, self.inv_adit(self.mult(q, t_curr)))
        g = r_prev
        x = s_prev
        y = t_prev

        if self.es_cero(g):
            return cero, cero, cero
        lider = g.coef_lider() 
        inv_lider = self.fp.inv_mult(lider)
        g_monico = self.mult_por_escalar(g, inv_lider)
        x_final = self.mult_por_escalar(x, inv_lider)
        y_final = self.mult_por_escalar(y, inv_lider)

        return g_monico, x_final, y_final
    
    def inv_mod(self, a, b):       # devuelve x tal que ax = 1 mod b
        g,x,y = self.gcd_ext(a, b)    
        if not self.es_uno(g):
            print(f"El polinomio ({a}) no tiene inverso multiplicativo módulo ({b})")
        else:  
            return x
            
    def pot_mod(self, a, k, b):    # a^k mod b
        if k < 0:
            raise ValueError('El exponente k no puede ser negativo.')
        resultado = self.uno()
        a = self.mod(a, b)
        if self.es_cero(a) and k > 0:
            return self.cero()
        while k > 0:
            if k % 2 == 1:
                resultado = self.mod(self.mult(resultado, a), b)
            a = self.mod(self.mult(a, a), b)
            k = k//2
        return resultado  
    
    def es_cero(self, a):          # a == 0
        return a.es_cero()
    
    def es_uno(self, a):           # a == 1
        return a.es_uno()
    
    def es_igual(self, a, b):      # a == b
        return a.__eq__(b)
    
    def es_irreducible(self, f):   # test de irreducibilidad de Rabin
        d = f.deg
        if d == 0:
            return False
        if d == 1:
            return True
        fact = f.factores()
        n = []
        for i in fact:
            n.append(d // i)
        x = self.elem_de_str(self.var)
        for i in n:
            exp = pow(self.fp.p,i)
            x_p_ni = self.pot_mod(x, exp, f)
            h = self.suma(x_p_ni, self.inv_adit(x))
            g = self.gcd(f, h)
            if not self.es_uno(g):
                return False
        exp_final = pow(self.fp.p, d)
        x_p_n = self.pot_mod(x, exp_final, f)
        g_final = self.suma(x_p_n, self.inv_adit(x))
        
        if self.es_cero(self.mod(g_final, f)):
            return True
        else:
            return False
        
    def _polinomio_aleatorio(self, grado_max):
        
        coefs = []
        for _ in range(grado_max + 1):
            coefs.append(self.fp.aleatorio())

        return Polinomio(self.fp, coefs, self.var)
    
    def sqfree_fact_fpx(self, f):
        lider = f.coef_lider()
        inv_lider = self.fp.inv_mult(lider)
        f1 = self.mult_por_escalar(f, inv_lider)
        resultado = self.uno()
        s = 1
        while not f1.es_uno():
            j = 1
            g = self.div(f1,self.gcd(f1, f1.derivada()))
            while not g.es_uno():
                f1 = self.div(f1,g)
                h = self.gcd(f1,g)
                m = self.div(g,h)
                if not m.es_uno(): resultado = self.mult(resultado, m)
                g = h
                j +=1
            if not f1.es_uno(): 
                p = self.fp.p
                print(f'El polinomio es una potencia {p}-ésima.')
                s = s*p
                t = []
                for indice, c in enumerate(f1):
                    if indice % p == 0:
                        t.append(c)
                f1 = Polinomio(self.fp, t, f.var)
        
        return resultado
    
    def didegr_fact_fpx(self, g):
        fpx = self
        L = {}
        x = fpx.elem_de_str(fpx.var)
        h = fpx.mod(x,g)
        k = 1
        while not g.es_uno():
            h = fpx.pot_mod(h,fpx.fp.p,g)
            f = fpx.gcd(h-x,g)
            if not f.es_uno():
                L[k] = f
                g = fpx.div(g,f)
                h = fpx.mod(h,g)
            k +=1
        if not L:
            return []
        grado_max = max(L.keys())
        resultado = []
        for i in range (1, grado_max +1):
            factor = L.get(i,fpx.uno())
            resultado.append(factor)
        return resultado
    
    def eqdegr_fact_fpx(self, r, h):
        grado_total = self.grado(h)
        if grado_total < r:
            return []
        num_factores = grado_total // r
        if num_factores == 1:
            return [h]
        H = [h]
        H_final = []
        while H:
            H_nuevo = []
            for h_actual in H:
                grado_max_alpha = self.grado(h_actual) - 1
                alpha = self._polinomio_aleatorio(grado_max_alpha)
                exponente = (pow(self.fp.p, r) - 1) // 2
                S = self.pot_mod(alpha, exponente, h_actual)
                S_menos_1 = self.suma(S, self.inv_adit(self.uno()))
                d = self.gcd(S_menos_1, h_actual)
                if self.es_uno(d) or self.es_igual(d, h_actual):
                    H_nuevo.append(h_actual)
                else:
                    d_complemento = self.div(h_actual, d)
                    if self.grado(d) == r:
                        H_final.append(d)
                    else:
                        H_nuevo.append(d)
                        
                    if self.grado(d_complemento) == r:
                        H_final.append(d_complemento)
                    else:
                        H_nuevo.append(d_complemento)
            H = H_nuevo
            if len(H_final) == num_factores:
                break
        return H_final
    
    def multiplicidad_fpx(self, f, u):
        e = 0
        f_actual = f 
        while True:
            q, r = self.divmod(f_actual, u)
            if self.es_cero(r):
                e += 1
                f_actual = q
            else:
                break
        return e
    
    def fact_fpx(self, f):                     
        g = self.sqfree_fact_fpx(f)
        h = self.didegr_fact_fpx(g)
        irreducibles = []
        for r in range(len(h)):
            if self.grado(h[r]) > 0:
                irreducibles += self.eqdegr_fact_fpx(r+1, h[r])
        factorizacion = []
        for u in irreducibles:
            e = self.multiplicidad_fpx(f, u)
            factorizacion += [(u,e)]
        return factorizacion
    
class ElementoFq:
    def __init__(self, fp, f, g, var = 'a'):
        self.fp = fp
        if f.var != g.var:
            raise ValueError('Los polinomios han de tener la misma variable.')
        self.var = var
        self.p = fp.p
        self.q = fp.p**g.deg
        self.repre = anillo_fp_x(fp, g.var).mod(f, g)
        self.modulo = g
        
    def mismo_Fq(self, other):
        if self.q != other.q or self.modulo.__ne__(other.modulo) :
            raise ValueError("Los elementos no pertenecen al mismo cuerpo Fq.")    

    def __eq__(self, other):
        if isinstance(other, ElementoFq):
            self.mismo_Fq(other)
            return self.repre.__eq__(other.repre)
        
        if isinstance(other, int):
            if other == 0:
                return self.es_cero()
            if other == 1:
                return self.es_uno()
            R_anillo = anillo_fp_x(self.fp, self.modulo.var)
            pol_constante = R_anillo.elem_de_int(other) 
            return self.repre.__eq__(pol_constante)
        
        return NotImplemented
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __neg__(self):
        return ElementoFq(self.fp, -self.repre, self.modulo, self.var)
    
    def __add__(self,other):
        self.mismo_Fq(other)
        return ElementoFq(self.fp, self.repre+other.repre, self.modulo, self.var)
    
    def __sub__(self, other):
        self.mismo_Fq(other)
        return ElementoFq(self.fp, self.repre -other.repre, self.modulo, self.var)
    
    def __mul__(self,other):
        self.mismo_Fq(other)
        return ElementoFq(self.fp, self.repre*other.repre, self.modulo, self.var)
    
    def __pow__(self, k):
        return ElementoFq(self.fp, anillo_fp_x(self.fp, self.modulo.var).pot_mod(self.repre, k, self.modulo), self.modulo, self.var)
    
    def __str__(self):
        return str(Polinomio(self.fp,self.repre.coef,self.var))
    
    def __repr__(self):
        return str(self)
    
    def es_cero(self):
        return self.repre.es_cero()
    
    def es_uno(self):
        return self.repre.es_uno()
    
    def inv_mult(self):         # a^(-1)
        if self.es_cero() :
            raise ZeroDivisionError("Inverso multiplicativo de 0 no existe en un cuerpo.")
        R = anillo_fp_x(self.fp, self.modulo.var)
        inv_poli = R.inv_mod(self.repre, self.modulo)
        if inv_poli is None:
             raise ValueError(f"El elemento {self} no tiene inverso (¿es el módulo irreducible?)")
        return ElementoFq(self.fp, inv_poli, self.modulo, self.var)
    
class cuerpo_fq:
    def __init__(self, fp, g, var='a'): # construye el cuerpo Fp[var]/<g(var)>
        self.caracteristica = fp.p
        self.var = var
        self.fp = fp
        self.q = fp.p**g.deg
        self.modulo = g

    def cero(self):                # 0
        return ElementoFq(self.fp, Polinomio.desde_entero(self.fp, 0, self.modulo.var), self.modulo, self.var)
    
    def uno(self):                 # 1
        return ElementoFq(self.fp, Polinomio.desde_entero(self.fp, 1, self.modulo.var), self.modulo, self.var)
    
    def elem_de_tuple(self, a):    # fabrica elemento a partir de tupla de coeficientes
        return ElementoFq(self.fp, Polinomio.desde_tuple(self.fp, a, self.modulo.var), self.modulo, self.var)
    
    def elem_de_int(self, a):      # fabrica elemento a partir de entero
        return ElementoFq(self.fp, Polinomio.desde_entero(self.fp,a,self.modulo.var), self.modulo, self.var)
    
    def elem_de_str(self, s):      # fabrica elemento parseando string
        return ElementoFq(self.fp, Polinomio.desde_str(self.fp, s, self.modulo.var), self.modulo, self.var)
    
    def conv_a_tuple(self, a):     # devuelve tupla de coeficientes sin ceros "extra"
        if isinstance(a, Polinomio):
            b = ElementoFq(self.fp, a, self.modulo, self.var)
        elif not isinstance(a, ElementoFq):
            return NotImplemented
        else:
            b = a
        return tuple(b.repre.coef)
    
    def conv_a_int(self, a):       # devuelve el entero correspondiente
        if isinstance(a, Polinomio):
            b = ElementoFq(self.fp, a, self.modulo, self.var)
        elif not isinstance(a, ElementoFq):
            return NotImplemented
        else:
            b = a
        return int(b.repre)
    
    def conv_a_str(self, a):       # pretty-printer
        if isinstance(a, Polinomio):
            b = ElementoFq(self.fp, a, self.modulo, self.var)
        elif not isinstance(a, ElementoFq):
            return NotImplemented
        else:
            b = a
        return str(b)
    
    def suma(self, a, b):          # a+b
        return a+b
    
    def inv_adit(self, a):         # -a
        return -a
    
    def mult(self, a, b):          # a*b
        return a*b
    
    def pot(self, a, k):           # a^k (k entero)
        return a**k
    
    def inv_mult(self, a):         # a^(-1)
        return a.inv_mult()
    
    def es_cero(self, a):          # a == 0
        return a.es_cero()
    
    def es_uno(self, a):           # a == 1
        return a.es_uno()
    
    def es_igual(self, a, b):      # a == b
        return a == b
    
    def aleatorio(self):           # devuelve un elemento aleatorio con prob uniforme
        n = self.modulo.deg 
        Cn = [self.fp.aleatorio() for i in range(0,n)]
        return ElementoFq(self.fp, Polinomio(self.fp, Cn, self.modulo.var), self.modulo, self.var)
    
    def tabla_suma(self):          # matriz de qxq correspondiente a la suma (con la notación int)
        print(f"Tabla de suma en Fq (q={self.q}):")
        print("     ", end="")
        for j in range(self.q):
            print(f"{j:4}", end="")
        print("\n" + "-" * (6 + 4 * self.q))
        for i in range(self.q):
            print(f"{i:2} |", end="")
            a = self.elem_de_int(i)
            for j in range(self.q):
                b = self.elem_de_int(j)
                resultado = self.suma(a, b)
                print(f"{self.conv_a_int(resultado):4}", end="")
            print()

    def tabla_mult(self):          # matriz de qxq correspondiente a la mult (con la notación int)
        print(f"Tabla de multiplicación en Fq (q={self.q}):")
        print("     ", end="")
        for j in range(self.q):
            print(f"{j:4}", end="")
        print("\n" + "-" * (6 + 4 * self.q))
        for i in range(self.q):
            print(f"{i:2} |", end="") 
            a = self.elem_de_int(i)
            for j in range(self.q):
                b = self.elem_de_int(j)
                resultado = self.mult(a, b)
                print(f"{self.conv_a_int(resultado):4}", end="")
            print()

    def tabla_inv_adit(self):
        inversos = []
        print(f"Inversos aditivos en Fq (q={self.q}):")
        for i in range(self.q):
            a = self.elem_de_int(i)
            inv = self.inv_adit(a)
            inv_int = self.conv_a_int(inv)
            inversos.append(inv_int)
            print(f"{i} -> {inv_int}")
        return inversos

    def tabla_inv_mult(self):      # lista de inv_mult (con la notación int)
        inversos = []
        print(f"Inversos multiplicativos en Fq (q={self.q}):")

        print("0 -> * (no tiene inverso)")
        inversos.append("*") 

        for i in range(1, self.q):
            a = self.elem_de_int(i)
            try:
                inv = self.inv_mult(a) 
                inv_int = self.conv_a_int(inv)
                inversos.append(inv_int)
                print(f"{i} -> {inv_int}")
            except (ZeroDivisionError, ValueError) as e:
                print(f"{i} -> ERROR ({e})")
                inversos.append(None)
        return inversos
           
    def cuadrado_latino(self, a):  # cuadrado latino para a != 0 (con notación int)
        if self.es_cero(a):
            raise ValueError("a no puede ser 0 para formar un cuadrado latino.")
        a_int = self.conv_a_int(a)
        print(f"Cuadrado latino generado con a = {a_int} (pol: {a}) en Fq (q={self.q}):\n")
        cuadrado = []
        print("     ", end="")
        for j in range(self.q):
            print(f"{j:4}", end="")
        print("\n" + "-" * (6 + 4 * self.q))
        for i in range(self.q):
            fila = []
            print(f"{i:2} |", end="")
            elem_i = self.elem_de_int(i)
            for j in range(self.q):
                elem_j = self.elem_de_int(j)
                valor = self.suma(self.mult(a, elem_i), elem_j)
                valor_int = self.conv_a_int(valor)
                fila.append(valor_int)
                print(f"{valor_int:4}", end="")
            cuadrado.append(fila)
            print()
        return cuadrado

class PolinomioFq:
    def __init__(self, fq, Cn, var = 'x'):
        if var == fq.var:
            raise ValueError ('La variable del anillo ha de ser distinta a la del cuerpo Fq.')
        self.var = var
        self.fq = fq
        self.caracteristica = fq.caracteristica
        self.coef = self._normalizar(Cn[:])
        self.deg = len(self.coef)-1
    
    def _normalizar(self, Cn):
        
        while len(Cn) > 1 and self.fq.es_cero(Cn[-1]):
            Cn.pop()
        if not Cn:
            Cn = [self.fq.cero()]
        return Cn
    
    @classmethod
    
    def desde_tuple(cls, fq, t, var = 'x'): # Suponemos que los elementos de la tupla son ElementoFq
        coefs = [c for c in t]
        return cls(fq, coefs, var)
    
    @classmethod
    
    def desde_int(cls, fq, n, var = 'x'):
        if n == 0:
            return cls(fq, [fq.cero()], var)
        coefs = []
        q = fq.q
        while n > 0:
            coefs.append(n % q)
            n = n // q
        
        Cn = [fq.elem_de_int(c) for c in coefs]
        return cls(fq, Cn, var)
    
    @classmethod
    
    def desde_str(cls, fq, s, var = 'x'):

        s = s.replace(" ", "")
        if not s:
            return cls(fq,[fq.cero()],var)
        
        if s[0] not in "+-":
            s = "+" + s
        terminos = []
        start_index = 0
        paren_depth = 0    
        
        for i, char in enumerate(s):
            if i == 0: continue 

            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
                if paren_depth < 0:
                    raise ValueError(f"Paréntesis desbalanceados en: {s}")
            elif char in "+-" and paren_depth == 0:
                terminos.append(s[start_index:i])
                start_index = i
        
        if paren_depth != 0:
             raise ValueError(f"Paréntesis desbalanceados en: {s}")
             
        terminos.append(s[start_index:])
        
        coef_dict = {} 
        for t in terminos:           
            sign_char = t[0]
            term_str = t[1:] 
            exp = 0
            if var not in term_str:
                coef_str_raw = term_str
                exp = 0
            else:
                parts = term_str.split(var, 1) 
                coef_str_raw = parts[0]
                exp_str = parts[1]
                
                if not exp_str:
                    exp = 1
                elif exp_str.startswith('^'):
                    exp = int(exp_str[1:])
                elif exp_str.startswith('**'):
                    exp = int(exp_str[2:])
                else:
                    raise ValueError(f"Formato de término no válido: {t}")
            if coef_str_raw.endswith('*'):
                coef_str = coef_str_raw[:-1]
            else:
                coef_str = coef_str_raw
            
            coef = fq.cero() 
            if coef_str == "":
                coef = fq.uno()
            elif coef_str.startswith('(') and coef_str.endswith(')'):
                coef = fq.elem_de_str(coef_str[1:-1])
            else:
                coef = fq.elem_de_str(coef_str)

            if sign_char == '-':
                coef = -coef 

            if exp in coef_dict:
                coef_dict[exp] = fq.suma(coef_dict[exp], coef)
            else:
                coef_dict[exp] = coef

        if not coef_dict:
            return cls(fq, [fq.cero()], var)
            
        grado_max = max(coef_dict.keys())
        coefs = [coef_dict.get(i, fq.cero()) for i in range(grado_max + 1)]
        
        return cls(fq, coefs, var)

    def __str__(self):
        s = []
        for i, a in enumerate(self.coef):
            if self.fq.es_cero(a):
                continue
            coef_str = str(a)
            needs_parens = " " in coef_str.strip()
            if needs_parens and coef_str.startswith("(") and coef_str.endswith(")"):
                needs_parens = False         
            formatted_coef = f"({coef_str})" if needs_parens else coef_str
            term = ""
            if i == 0:
                term = formatted_coef
            elif i == 1:
                if formatted_coef == "1": term = self.var
                elif formatted_coef == "-1": term = f"-{self.var}"
                else: term = f"{formatted_coef}{self.var}"
            else: 
                if formatted_coef == "1": term = f"{self.var}^{i}"
                elif formatted_coef == "-1": term = f"-{self.var}^{i}"
                else: term = f"{formatted_coef}{self.var}^{i}"
            s.append(term)
        if not s:
            return "0"
        return " + ".join(reversed(s))

    def __iter__(self):
        return iter(self.coef)

    def __int__(self):
        resultado = 0
        for i in range(self.deg,-1,-1):
            resultado = resultado*self.fq.q+self.fq.conv_a_int(self.coef[i])
        return resultado

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if self.fq != other.fq:
            raise ValueError('Los polinmios han de tener sus coeficientes en el mismo cuerpo Fq.')
        Fq = self.fq
        grado_max = max(len(self), len(other))
        suma_coef = [Fq.cero() for _ in range(grado_max)]

        for i, coef in enumerate(self):
            suma_coef[i] = suma_coef[i] + coef
        
        for i, coef in enumerate(other):
            suma_coef[i] = suma_coef[i] + coef
            
        return PolinomioFq(Fq, suma_coef, self.var)

    def __len__(self):
        return len(self.coef)

    def __getitem__(self, i):
        return self.coef[i]
    
    def __bool__(self):
        return not self.es_cero()

    def __neg__(self):
        inverso = [-c for c in self]
        return PolinomioFq(self.fq, inverso, self.var)
    
    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __mul__(self, other):
        if self.fq != other.fq:
            raise ValueError("Los polinomios deben pertenecer al mismo cuerpo Fq.")
        elif self.var != other.var:
            raise ValueError('Los polinomios han de tener la misma variable muda.')        
        Fq = self.fq
        var = self.var       
        if self.es_cero() or other.es_cero():
            return PolinomioFq(Fq, [Fq.cero()], var)        
        m = self.deg
        n = other.deg
        nuevo_coef = [Fq.cero() for _ in range(m + n + 1)]
        for i in range(m+1):
            for j in range(n+1):
                nuevo_coef[i + j] = nuevo_coef[i + j] + (self.coef[i] * other.coef[j])
        
        return PolinomioFq(self.fq, nuevo_coef, self.var)
    
    def divmod(self, other):

        if self.fq != other.fq:
            raise ValueError("Los polinomios deben pertenecer al mismo cuerpo Fq")
        elif self.var != other.var:
            raise ValueError('Los polinomios han de tener la misma variable muda.')
     
        Fq = self.fq
        var = self.var
    
        if other.es_cero():
            raise ZeroDivisionError("División por polinomio nulo")
    
        degA = self.deg
        degB = other.deg
    
        if degA < degB:
            cero_pol = PolinomioFq(Fq, [Fq.cero()], var)
            return (cero_pol, self)            
    
        Q = [Fq.cero() for _ in range(degA - degB + 1)]
        R_coef = self.coef[:] 
        inv_lider_B = Fq.inv_mult(other.coef_lider())
    
        for k in range(degA - degB, -1, -1):
            coeff = R_coef[degB + k] * inv_lider_B
            Q[k] = coeff
    
            for j in range(degB + 1):
                R_coef[j + k] = R_coef[j + k] - coeff * other[j]
    
        return PolinomioFq(Fq, Q, var), PolinomioFq(Fq, R_coef, var)
    
    def __floordiv__(self, other):

        Q, _ = self.divmod(other)
        return Q
    
    def __mod__(self, other):

        _, R = self.divmod(other)
        return R
    
    def __eq__(self, other):
        if isinstance(other, PolinomioFq):
            return (self.fq == other.fq and
                    self.var == other.var and
                    self.coef == other.coef)
        
        if isinstance(other, int):
            if other == 0:
                return self.es_cero()
            if other == 1:
                return self.es_uno()
            else: 
                return self == PolinomioFq.desde_int(self.fq, other, self.var)
        
        if isinstance(other, ElementoFq): 
            return self == PolinomioFq(self.fq, [other], self.var)
        
        return NotImplemented
    
    def es_cero(self):
        return self.deg == 0 and self.fq.es_cero(self.coef[0])
    
    def es_uno(self):
        return self.deg == 0 and self.fq.es_uno(self.coef[0])
    
    def coef_lider(self):
        return self.coef[-1]
    
    def factores(self):   
        d = self.deg      
        if d <= 1:
            return []  
        factores = []
        if d % 2 == 0:
            factores.append(2)
            while d % 2 == 0:
                d //= 2 
        p = 3
        while p * p <= d:
            if d % p == 0:
                factores.append(p)
                while d % p == 0:
                    d //= p
            p += 2         
        if d > 1:
            factores.append(d)    
        return factores

    def derivada(self):
        if self.deg == 0:
            return PolinomioFq(self.fq, [self.fq.cero()], self.var)

        nuevos_coef = [self.fq.cero()] * self.deg 

        for i in range(1, self.deg + 1):

            nuevos_coef[i-1] = self.coef[i] * self.fq.elem_de_int(i)
            
        return PolinomioFq(self.fq, nuevos_coef, self.var)
class anillo_fq_x:
    
    def __init__(self, fq, var = 'x'):
        if var == fq.var:
            raise ValueError ('La variable del anillo ha de ser distinta a la del cuerpo Fq.')
        self.fq = fq
        self.caracteristica = fq.fp
        self.var = var
        
        
    def cero(self):     
        return PolinomioFq(self.fq, [self.fq.cero()], self.var)
    
    def uno(self):                 # 1
        return PolinomioFq(self.fq, [self.fq.uno()], self.var)
    
    def elem_de_tuple(self, a):
        return PolinomioFq.desde_tuple(self.fq, a, self.var)
        
    def elem_de_int(self, a):
        return PolinomioFq.desde_int(self.fq, a, self.var)
        
    def elem_de_str(self, s):
        return PolinomioFq.desde_str(self.fq, s, self.var)
        
    def conv_a_tuple(self, a):
        return tuple(a)
    
    def conv_a_int(self, a):
        return int(a)
    
    def conv_a_str(self, a):
        return str(a)
    
    def suma(self, a, b):
        return a+b
    
    def inv_adit(self, a):
        return -a
    
    def mult(self, a, b):
        if a.deg == 0:
            return self.mult_por_escalar(b, int(a))
        elif b.deg == 0:
            return self.mult_por_escalar(a, int(b))
        return a*b
    
    def mult_por_escalar(self, a, e):
        if isinstance(e, int):
            n = self.fq.elem_de_int(e)
        elif isinstance(e, str):
            n = self.fq.elem_de_str(e)
        elif isinstance(e, ElementoFq):
            n = e
        else:
             raise TypeError(f"Tipo no válido para multiplicación escalar: {type(e)}")
        Cn = [n*c for c in a]
        return PolinomioFq(self.fq, Cn, self.var)
    
    def divmod(self, a, b):
        return a.divmod(b)
    
    def div(self, a, b):
        return a//b
    
    def mod(self, a, b):
        return a%b
    
    def grado(self, a):
        return a.deg
    
    def gcd(self, a, b):
        while not self.es_cero(b):
            a, b = b, self.mod(a, b)
        if self.es_cero(a):
            return self.cero()
        
        lider = a.coef_lider()
        inverso_lider = self.fq.inv_mult(lider)
        return self.mult_por_escalar(a, inverso_lider)
    
    def gcd_ext(self, a, b):
        cero = self.cero()
        uno = self.uno()
        r_prev, r_curr = a, b
        s_prev, s_curr = uno, cero
        t_prev, t_curr = cero, uno

        while not self.es_cero(r_curr):
            q, r_nuevo = self.divmod(r_prev, r_curr)
            r_prev, r_curr = r_curr, r_nuevo
            s_prev, s_curr = s_curr, self.suma(s_prev, self.inv_adit(self.mult(q, s_curr)))
            t_prev, t_curr = t_curr, self.suma(t_prev, self.inv_adit(self.mult(q, t_curr)))
        g = r_prev
        x = s_prev
        y = t_prev

        if self.es_cero(g):
            return cero, cero, cero
        lider = g.coef_lider() 
        inv_lider = self.fq.inv_mult(lider)
        g_monico = self.mult_por_escalar(g, inv_lider)
        x_final = self.mult_por_escalar(x, inv_lider)
        y_final = self.mult_por_escalar(y, inv_lider)

        return g_monico, x_final, y_final
        
    def inv_mod(self, a, b):
        g,x,y = self.gcd_ext(a, b)    
        if not self.es_uno(g):
            print(f"El polinomio ({a}) no tiene inverso multiplicativo módulo ({b})")
        else:  
            return x
        
    def pot_mod(self, a, k, b):
        if k < 0:
            raise ValueError('El exponente k no puede ser negativo.')
        resultado = self.uno()
        a = self.mod(a, b)
        if self.es_cero(a) and k > 0:
            return self.cero()
        while k > 0:
            if k % 2 == 1:
                resultado = self.mod(self.mult(resultado, a), b)
            a = self.mod(self.mult(a, a), b)
            k = k//2
        return resultado  
    
    def es_cero(self, a):
        return a.es_cero()
    
    def es_uno(self, a):
        return a.es_uno()
    
    def es_igual(self, a, b):
        return a == b
    
    def es_irreducible(self, f):
        d = f.deg
        if d == 0:
            return False
        if d == 1:
            return True
        
        fact = f.factores() 
        n = []
        for i in fact:
            n.append(d // i)        
        x = self.elem_de_str(self.var)
        q = self.fq.q     
        for i in n:
            exp = pow(q, i)  
            x_q_ni = self.pot_mod(x, exp, f)
            h = self.suma(x_q_ni, self.inv_adit(x))
            g = self.gcd(f, h)
            if not self.es_uno(g):
                return False
        exp_final = pow(q, d) 
        x_q_n = self.pot_mod(x, exp_final, f)
        g_final = self.suma(x_q_n, self.inv_adit(x))  
        if self.es_cero(self.mod(g_final, f)):
            return True
        else:
            return False
    
    def _polinomio_aleatorio(self, grado_max):
        
        coefs = []
        for _ in range(grado_max + 1):
            coefs.append(self.fq.aleatorio())

        return PolinomioFq(self.fq, coefs, self.var)
    
    def sqfree_fact_fqx(self, f):
        lider = f.coef_lider()
        inv_lider = self.fq.inv_mult(lider)
        f1 = self.mult_por_escalar(f, inv_lider)
        resultado = self.uno()
        s = 1
        while not f1.es_uno():
            j = 1
            g = self.div(f1,self.gcd(f1, f1.derivada()))
            while not g.es_uno():
                f1 = self.div(f1,g)
                h = self.gcd(f1,g)
                m = self.div(g,h)
                if not m.es_uno(): resultado = self.mult(resultado, m)
                g = h
                j +=1
            if not f1.es_uno(): 
                p = self.caracteristica.p
                q = self.fq.q
                
                print(f'El polinomio es una potencia {p}-ésima.')
                s = s*p
                raiz = q//p
                t = []
                for indice, c in enumerate(f1):
                    if indice % p == 0:
                        c_raiz = self.fq.pot(c, raiz)
                        t.append(c_raiz)
                f1 = PolinomioFq(self.fq, t, f.var)  
        return resultado
    
    def didegr_fact_fqx(self, g):
        L = {}
        x = self.elem_de_str(self.var)
        h = self.mod(x,g)
        k = 1
        while not g.es_uno():
            h = self.pot_mod(h,self.fq.q,g)
            f = self.gcd(h-x,g)
            if not f.es_uno():
                L[k] = f
                g = self.div(g,f)
                h = self.mod(h,g)
            k +=1
        if not L:
            return []
        grado_max = max(L.keys())
        resultado = []
        for i in range (1, grado_max +1):
            factor = L.get(i,self.uno())
            resultado.append(factor)
        return resultado
    
    def eqdegr_fact_fqx(self, r, h):
        grado_total = self.grado(h)
        if grado_total < r:
            return []
        num_factores = grado_total // r
        if num_factores == 1:
            return [h]
        H = [h]
        H_final = []
        while H:
            H_nuevo = []
            for h_actual in H:
                grado_max_alpha = self.grado(h_actual) - 1
                alpha = self._polinomio_aleatorio(grado_max_alpha)
                exponente = (pow(self.fq.q, r) - 1) // 2
                S = self.pot_mod(alpha, exponente, h_actual)
                S_menos_1 = self.suma(S, self.inv_adit(self.uno()))
                d = self.gcd(S_menos_1, h_actual)
                if self.es_uno(d) or self.es_igual(d, h_actual):
                    H_nuevo.append(h_actual)
                else:
                    d_complemento = self.div(h_actual, d)
                    if self.grado(d) == r:
                        H_final.append(d)
                    else:
                        H_nuevo.append(d)
                        
                    if self.grado(d_complemento) == r:
                        H_final.append(d_complemento)
                    else:
                        H_nuevo.append(d_complemento)
            H = H_nuevo
            if len(H_final) == num_factores:
                break
        return H_final
    
    def multiplicidad_fqx(self, f, u):
        e = 0
        f_actual = f 
        while True:
            q, r = self.divmod(f_actual, u)
            if self.es_cero(r):
                e += 1
                f_actual = q
            else:
                break
        return e
    
    def fact_fqx(self, f):                   
        g = self.sqfree_fact_fqx(f)
        h = self.didegr_fact_fqx(g)
        irreducibles = []
        for r in range(len(h)):
            if self.grado(h[r]) > 0:
                irreducibles += self.eqdegr_fact_fqx(r+1, h[r])
        factorizacion = []
        for u in irreducibles:
            e = self.multiplicidad_fqx(f, u)
            factorizacion += [(u,e)]
        return factorizacion

    