import random
#Nos dice si n es un resto cuadrático para 
#p primo impar por el Criterio de Euler

def es_resto_cuad(c,p):
    if c%p == 0: return True
    
    return pow(c,(p-1)//2,p) == 1

def tonelli_shanks(c, p):
    
    if c%p == 0: return 0
    
    if not es_resto_cuad(c, p): 
        print('Este valor no es un resto cuaddrático.')
        return None
    
    else:
        print('Este valor es un resto cuadrático.')
        
    if p%4 == 3:
        return pow(c, (p+1)//4,p)
    if p % 8 == 5:
        
        c_0 = pow(c, (p-1)//4, p)
        
        if c_0 == 1:
            return pow(c, (p+3)//8,p)
        else:
            return 2*c*pow(4*c,(p-5)//8,p) % p
    
    q = p-1
    s = 0
    while q%2 ==0:
        s+=1
        q//=2
    print('q =', q)
    print('s =',s)
    z = 2
    #Vamos a buscar el generador. Si supiéramos que 
    #el p es pequeño, casi que mejor buscarlo con un bucle...
    while es_resto_cuad(z, p):
        z = random.randint(2, p-1)
    print('z=',z)
        
    #Vamos a inicializar
    y = z
    r = s
    x = pow(c,(q-1)//2,p)
    b = c*pow(x,2,p) % p
    x = c*x % p
    print('y=', y)
    print('r=',r)
    print('b=',b)
    print('x=', x)
    i = 1
    while b % p != 1:
        print('Iteración', i)
        m = 1
        base = pow(b,2,p)
        prod = base
        while prod % p != 1:
            m +=1
            prod = prod *base
        if r == m:
            print('Este valor no es un resto cuadrático.')
            return None
        t = pow(y, 2**(r-m-1), p)
        y = t**2
        r = m
        x = x*t % p
        b = b*y % p
        i +=1
        print('y=',y)
        print('r=',r)
        print('b=',b)
        print('x=', x)
    return x
    
    
    
    