# para esta actividad necesitamos la anterior (cuerpos_finitos.py)
# completa, pero quitando las funciones de factorizaci√≥n en anillo_fp_x
# y en anillo_fq_x, ya que las implementaremos aqu√≠ (no se olviden de
# quitarlas)
import cuerpos_finitos as cf

# square-free factorization
# input: fpx --> anillo_fp_x
# input: f --> polinomio fabricado por fpx (objeto opaco) no nulo
# output: g = producto de los factores irreducibles m√≥nicos de f, es decir,
# si f = c * f1^e1 * f2^e2 * ... * fr^er con los fi irreducibles m√≥nicos
# distintos entre si, ei >= 1, c en fp, entonces g = f1 * f2 * ... * fr
def sqfree_fact_fpx(fpx, f):
    lider = f.coef_lider()
    inv_lider = fpx.fp.inv_mult(lider)
    f1 = fpx.mult_por_escalar(f, inv_lider)
    resultado = fpx.uno()
    s = 1
    while not f1.es_uno():
        j = 1
        g = fpx.div(f1,fpx.gcd(f1, f1.derivada()))
        while not g.es_uno():
            f1 = fpx.div(f1,g)
            h = fpx.gcd(f1,g)
            m = fpx.div(g,h)
            if not m.es_uno(): resultado = fpx.mult(resultado, m)
            g = h
            j +=1
        if not f1.es_uno(): 
            p = fpx.fp.p
            print(f'El polinomio es una potencia {p}-ésima.')
            s = s*p
            t = []
            for indice, c in enumerate(f1):
                if indice % p == 0:
                    t.append(c)
            f1 = cf.Polinomio(fpx.fp, t, f.var)
    
    return resultado

# distinct-degree factorization
# input: fpx --> anillo_fp_x
# input: g --> polinomio de fpx (objeto opaco) que es producto de factores
# irreducibles m√≥nicos distintos cada uno con multiplicidad uno
# output: [h1, h2, ..., hr], donde hi = producto de los factores irreducibles
# m√≥nicos de h de grado = i, el √∫ltimo hr debe ser no nulo y por supuesto
# g = h1 * h2 * ... * hr
def didegr_fact_fpx(fpx, g):
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
# equal-degree factorization
# input: fpx --> anillo_fp_x (supondremos p impar)
# input: r --> int
# input: h --> polinomio de fpx (objeto opaco) que es producto de factores
# irreducibles m√≥nicos distintos de grado r con multiplicidad uno
# output: [u1, ..., us], donde h = u1 * u2* ... * us y los ui son irreducibles
# m√≥nicos de grado = r
def eqdegr_fact_fpx(fpx, r, h):
    grado_total = fpx.grado(h)
    if grado_total < r:
        return []
    num_factores = grado_total // r
    if num_factores == 1:
        return [h]
    H = [h]
    H_final = []
    while len(H_final) != num_factores and H:
        H_nuevo = []
        for h_actual in H:
            grado_max_alpha = fpx.grado(h_actual) - 1
            alpha = fpx._polinomio_aleatorio(grado_max_alpha)
            exponente = (pow(fpx.fp.p, r) - 1) // 2
            S = fpx.pot_mod(alpha, exponente, h_actual)
            S_menos_1 = fpx.suma(S, fpx.inv_adit(fpx.uno()))
            d = fpx.gcd(S_menos_1, h_actual)
            if fpx.es_uno(d) or fpx.es_igual(d, h_actual):
                H_nuevo.append(h_actual)
            else:
                d_complemento = fpx.div(h_actual, d)
                if fpx.grado(d) == r:
                    H_final.append(d)
                else:
                    H_nuevo.append(d)
                    
                if fpx.grado(d_complemento) == r:
                    H_final.append(d_complemento)
                else:
                    H_nuevo.append(d_complemento)
        H = H_nuevo
    return H_final
# multiplicidad de factor irreducible m√≥nico
# input: fpx --> anillo_fp_x
# input: f --> polinomio de fpx (objeto opaco) no nulo
# input: u --> polinomio irreducible m√≥nico (objeto opaco) de grado >= 1
# output: multiplicidad de u como factor de f, es decir, el entero e >= 0
# mas grande tal que u^e | f
def multiplicidad_fpx(fpx, f, u):
    e = 0
    f_actual = f 
    while True:
        q, r = fpx.divmod(f_actual, u)
        if fpx.es_cero(r):
            e += 1
            f_actual = q
        else:
            break
    return e
# factorizaci√≥n de Cantor-Zassenhaus
# input: fpx --> anillo_fp_x (supondremos p impar)
# input: f --> polinomio de fpx (objeto opaco)
# output: [(f1,e1), ..., (fr,er)] donde f = c * f1^e1 * ... * fr^er es la
# factorizaci√≥n completa de f en irreducibles m√≥nicos fi con multiplicidad
# ei >= 1 y los fi son distintos entre si y por supuesto c es el coeficiente
# principal de f
def fact_fpx(fpx, f):                     # mantener esta implementaci√≥n
    g = sqfree_fact_fpx(fpx, f)
    h = didegr_fact_fpx(fpx, g)
    irreducibles = []
    for r in range(len(h)):
        if fpx.grado(h[r]) > 0:
            irreducibles += eqdegr_fact_fpx(fpx, r+1, h[r])
    factorizacion = []
    for u in irreducibles:
        e = multiplicidad_fpx(fpx, f, u)
        factorizacion += [(u,e)]
    return factorizacion

# esta linea es para a√±adir la funci√≥n de factorizaci√≥n de Cantor-Zassenhaus
# como un m√©todo de la clase anillo_fp_x
cf.anillo_fp_x.factorizar = fact_fpx

# square-free factorization
# input: fqx --> anillo_fq_x
# input: f --> polinomio fabricado por fqx (objeto opaco) no nulo
# output: g = producto de los factores irreducibles m√≥nicos de f, es decir,
# si f = c * f1^e1 * f2^e2 * ... * fr^er con los fi irreducibles m√≥nicos
# distintos entre si, ei >= 1, c en fq, entonces g = f1 * f2 * ... * fr
def sqfree_fact_fqx(fqx, f):
    lider = f.coef_lider()
    inv_lider = fqx.fq.inv_mult(lider)
    f1 = fqx.mult_por_escalar(f, inv_lider)
    resultado = fqx.uno()
    s = 1
    while not f1.es_uno():
        j = 1
        g = fqx.div(f1,fqx.gcd(f1, f1.derivada()))
        while not g.es_uno():
            f1 = fqx.div(f1,g)
            h = fqx.gcd(f1,g)
            m = fqx.div(g,h)
            if not m.es_uno(): resultado = fqx.mult(resultado, m)
            g = h
            j +=1
        if not f1.es_uno(): 
            p = fqx.caracteristica.p
            q = fqx.fq.q
            
            print(f'El polinomio es una potencia {p}-ésima.')
            s = s*p
            raiz = q//p
            t = []
            for indice, c in enumerate(f1):
                if indice % p == 0:
                    c_raiz = fqx.fq.pot(c, raiz)
                    t.append(c_raiz)
            f1 = cf.PolinomioFq(fqx.fq, t, f.var)  
    return resultado

# distinct-degree factorization
# input: fqx --> anillo_fq_x
# input: g --> polinomio de fqx (objeto opaco) que es producto de factores
# irreducibles m√≥nicos distintos cada uno con multiplicidad uno
# output: [h1, h2, ..., hr], donde hi = producto de los factores irreducibles
# m√≥nicos de h de grado = i, el √∫ltimo hr debe ser no nulo y por supuesto
# g = h1 * h2 * ... * hr
def didegr_fact_fqx(fqx, g):
    L = {}
    x = fqx.elem_de_str(fqx.var)
    h = fqx.mod(x,g)
    k = 1
    while not g.es_uno():
        h = fqx.pot_mod(h,fqx.fq.q,g)
        f = fqx.gcd(h-x,g)
        if not f.es_uno():
            L[k] = f
            g = fqx.div(g,f)
            h = fqx.mod(h,g)
        k +=1
    if not L:
        return []
    grado_max = max(L.keys())
    resultado = []
    for i in range (1, grado_max +1):
        factor = L.get(i,fqx.uno())
        resultado.append(factor)
    return resultado

# equal-degree factorization
# input: fqx --> anillo_fq_x (supondremos q impar)
# input: r --> int
# input: h --> polinomio de fqx (objeto opaco) que es producto de factores
# irreducibles m√≥nicos distintos de grado r con multiplicidad uno
# output: [u1, ..., us], donde h = u1 * u2* ... * us y los ui son irreducibles
# m√≥nicos de grado = r
def eqdegr_fact_fqx(fqx, r, h):
    grado_total = fqx.grado(h)
    if grado_total < r:
        return []
    num_factores = grado_total // r
    if num_factores == 1:
        return [h]
    H = [h]
    H_final = []
    while len(H_final) != num_factores and H:
        H_nuevo = []
        for h_actual in H:
            grado_max_alpha = fqx.grado(h_actual) - 1
            alpha = fqx._polinomio_aleatorio(grado_max_alpha)
            exponente = (pow(fqx.fq.q, r) - 1) // 2
            S = fqx.pot_mod(alpha, exponente, h_actual)
            S_menos_1 = fqx.suma(S, fqx.inv_adit(fqx.uno()))
            d = fqx.gcd(S_menos_1, h_actual)
            if fqx.es_uno(d) or fqx.es_igual(d, h_actual):
                H_nuevo.append(h_actual)
            else:
                d_complemento = fqx.div(h_actual, d)
                if fqx.grado(d) == r:
                    H_final.append(d)
                else:
                    H_nuevo.append(d)
                    
                if fqx.grado(d_complemento) == r:
                    H_final.append(d_complemento)
                else:
                    H_nuevo.append(d_complemento)
        H = H_nuevo
    return H_final

# multiplicidad de factor irreducible m√≥nico
# input: fqx --> anillo_fq_x
# input: f --> polinomio de fqx (objeto opaco) no nulo
# input: u --> polinomio irreducible m√≥nico (objeto opaco) de grado >= 1
# output: multiplicidad de u como factor de f, es decir, el entero e >= 0
# mas grande tal que u^e | f
def multiplicidad_fqx(fqx, f, u):
    e = 0
    f_actual = f 
    while True:
        q, r = fqx.divmod(f_actual, u)
        if fqx.es_cero(r):
            e += 1
            f_actual = q
        else:
            break
    return e

# factorizaci√≥n de Cantor-Zassenhaus
# input: fqx --> anillo_fq_x (supondremos q impar)
# input: f --> polinomio de fqx (objeto opaco)
# output: [(f1,e1), ..., (fr,er)] donde f = c * f1^e1 * ... * fr^er es la
# factorizaci√≥n completa de f en irreducibles m√≥nicos fi con multiplicidad
# ei >= 1 y los fi son distintos entre si y por supuesto c es el coeficiente
# principal de f
def fact_fqx(fqx, f):                     # mantener esta implementaci√≥n
    g = sqfree_fact_fqx(fqx, f)
    h = didegr_fact_fqx(fqx, g)
    irreducibles = []
    for r in range(len(h)):
        if fqx.grado(h[r]) > 0:
            irreducibles += eqdegr_fact_fqx(fqx, r+1, h[r])
    factorizacion = []
    for u in irreducibles:
        e = multiplicidad_fqx(fqx, f, u)
        factorizacion += [(u,e)]
    return factorizacion

# esta linea es para a√±adir la funci√≥n de factorizaci√≥n de Cantor-Zassenhaus
# como un m√©todo de la clase anillo_fq_x
cf.anillo_fq_x.factorizar = fact_fqx
