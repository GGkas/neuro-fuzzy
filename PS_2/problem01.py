import numpy as np
# Askisi 1

# f(x) = x1^2 + 5x2^2 -3x1x2
A=np.array([[2,-3],[-3,10]])

def grad_f(x):
    x1=x[0,0]
    x2=x[1,0]
    return np.array([[2*x1-3*x2],[10*x2-3*x1]])

def compute_a(g,p):
    # print("PARONOMASTHS: ", p.T.dot((A.dot(p))))
    numerator=-g.T.dot(p)
    if numerator==0:
        return np.array([[0]])
    else:
        denominator=p.T.dot((A.dot(p)))
        a=numerator/denominator
        return a

def step1(x0):
    g0=grad_f(x0)
    print("g0 = " + str(g0) + "\n")
    p0=-g0
    # p0=-grad_f(x0)
    a0=compute_a(g0,p0)
    x1=x0+a0[0,0]*p0
    # print("p0:",p0)
    # print("x1:", x1)
    return (x1,g0,p0)
def step(x,g_prev,p_prev):
    end=False
    # compute g
    g=grad_f(x)
    print("gradient:\n", g)
    # compute b
    # print("PARONOMASTHS:",g_prev.T.dot(p_prev))
    # print(g_prev.T)
    # print(p_prev)
    b=(g.T.dot(g))/(g_prev.T.dot(g_prev))
    print("b:", b)
    # compute p
    p=-g+b*p_prev
    print("p:\n",p)
    # compute a
    a=compute_a(g,p)
    print("a:",a[0,0])
    if a[0,0]==0:
        print("REACHED MINIMUM. END OF ALGORITHM.")
        end=True
    # compute new x
    x=x+a[0,0]*p
    if x[0,0] < 0.000001:
        x[0,0]=0
    if x[1,0] < 0.000001:
        x[1,0]=0
    print("new x:\n",x)
    return(x,g,p,end)
x0=np.array([[1],[1]])
print("x0:\n",x0)
(x1,g0,p0)=step1(x0)
print("x1:\n",x1)

(x,g_prev,p_prev)=(x1,g0,p0)
for i in range(10):
    print("\n\nIteration:",i+1)
    (x,g_prev,p_prev,end)=step(x,g_prev,p_prev)
    if end:
        print("Found minimun:\n",x)
        break
# print("2nd iteration:")
# (x2,g1,p1)=step(x1,g0,p0)
#
# print("3rd iteration:")
# (x3,g2,p2)=step(x2,g1,p1)
#
# print("4th iteration:")
# (x4,g3,p3)=step(x3,g2,p2)
#
# print("5th iteration:")
# (x5,g4,p4)=step(x4,g3,p3)
#
# print("6th iteration:")
# (x6,g5,p5)=step(x5,g4,p4)
#
# print("7th iteration:")
# (x7,g6,p6)=step(x6,g5,p5)