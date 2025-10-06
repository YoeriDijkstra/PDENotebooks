import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, clear_output
import matplotlib.animation as animation
import matplotlib
# matplotlib.rcParams['animation.embed_limit'] = 2**128

def Traveling_waves(L,d,c):
    
    display(Markdown('# The animation is being rendered, please be patient.'))
    
    # symbols, functions and accuracy initialisation
    t,x,n = sp.symbols('t,x,n')
    phi = sp.Function('phi')(t)
    f = sp.Function('f')(x)
    g = sp.Function('g')(x)
    h = sp.Function('h')(x,t)
    v = sp.Function('v')(x)
    V = sp.Function('V')(x,t)
    N = 25

    # constant definition
    g = 9.81
    nu = np.sqrt(g*d)
    eta = 0
    T = 25/2*3600
    omega = (2*sp.pi)/T

    # function definitions
    phi = sp.sin(omega*t)

    # find function V that satisfies nonhomogeneous time-dependent BC's
    ode_v = sp.diff(v,x,x)
    sol = sp.dsolve(ode_v)
    constants = sp.solve([sol.rhs.subs(x,0)-phi,sol.rhs.diff(x).subs(x,L)-eta],sp.symbols('C1 C2'))
    V = sol.rhs.subs(constants)

    # define new initial condition and right hand side
    f = -V.subs(t,0)
    g = -V.diff(t).subs(t,0)
    h = -V.diff(t).diff(t)+nu**2*V.diff(x).diff(x)-c*V.diff(t)

    # find the Fourier sine expansion of h(x,t) with respect to x
    hn = sp.Function('hn')(t,n)
    hn = sp.integrate(h*sp.sin((1+2*n)*sp.pi*x/(2*L)),(x,0,L))/L*2

    # find the Fourier sine expansion of f(x) and g(x) with respect to x
    fn = sp.Function('fn')(n)
    gn = sp.Function('gn')(n)
    fn = sp.integrate(f*sp.sin((1+2*n)*sp.pi*x/(2*L)),(x,0,L))/L*2
    gn = sp.integrate(g*sp.sin((1+2*n)*sp.pi*x/(2*L)),(x,0,L))/L*2

    # find the Fourier sine expansion of w(t,x) with respect to x and construct w
    s = sp.symbols('s')
    Ln = sp.Function('Ln')(n)
    Ln = -(sp.pi*(1+2*n)/(2*L))**2
    w = sp.Function('w')(x,t)
    w = 0

    for nn in np.arange(0,N+1):
        Lnn = Ln.subs(n,nn)
        hnn = hn.subs(n,nn)
        fnn = fn.subs(n,nn)
        gnn = gn.subs(n,nn)
        wnn = sp.Function('wnn')(t)
        ode_wnn = sp.diff(wnn,t,t)-Lnn*nu**2*wnn-hnn+c*sp.diff(wnn,t)
        sol = sp.dsolve(ode_wnn,wnn,hint='nth_linear_constant_coeff_undetermined_coefficients',simplify =False)
        constants = sp.solve([sol.rhs.subs(t,0)-fnn,sol.rhs.diff(t).subs(t,0)-gnn],sp.symbols('C1 C2'),check=False,simplify=False,rational=False)
        wnn = sol.rhs.subs(constants)
        w += wnn*sp.sin(sp.pi*(1+2*nn)*x/(2*L))

    # make finally u(x,t)
    u = sp.Function('u')(x,t)
    u = w + V

    # evaluate u(x,t) at several x and t and plot
    u_lam = sp.lambdify((x,t),u,modules=['numpy'])
    tvec = np.linspace(0,6*T,601)
    xvec = np.linspace(0,L,101)
    # repeat some lines so that the animation will be more clear
    tvec2 = []
    for i,t in enumerate(tvec):
        if t<=L/10:
            tvec2.append(t)
            tvec2.append(t)
            tvec2.append(t)
            tvec2.append(t)
        elif t<=2*L/10:
            tvec2.append(t)
            tvec2.append(t)
            tvec2.append(t)
        elif t<=3*L/10:
            tvec2.append(t)
            tvec2.append(t)
        else:
            tvec2.append(t)
    tvec = tvec2
    tv, xv = np.meshgrid(tvec, xvec)
    umat = u_lam(xv,tv)
    u_max = np.copy(umat[-1,:])
    for i,uL in enumerate(u_max):
        if i>0:
            if uL<u_max[i-1]:
                u_max[i] = u_max[i-1]
    u_min = np.copy(umat[-1,:])
    for i,uL in enumerate(u_min):
        if i>0:
            if uL>u_min[i-1]:
                u_min[i] = u_min[i-1]

    fig = plt.figure(1);
    index = int(L/1e4-4)
    index = np.max([np.min([index,4]),0])
    ylist = np.array([2.5,2.5,3,4,5])
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, L), ylim=(-ylist[index],ylist[index]));
    plt.yticks(np.arange(-ylist[index],ylist[index]+0.5,0.5));
    ax.grid();
    ax.set_ylabel('$\\xi$');
    ax.set_xlabel('$x$');

    line, = ax.plot([], [], '-', lw=2,label='$\\xi(x,t)$');
    maxline, = ax.plot([], [], '--', lw=2,label='$\\max_{\\tau\\leq t}\\xi(L,\\tau)$');
    minline, = ax.plot([], [], '--', lw=2,label='$\\min_{\\tau\\leq t}\\xi(L,\\tau)$');
    dot, = ax.plot([], [], 'o', lw=2,label='$\\phi(t)$');
    time_template = '$t$ = %1.1fs';
    time_text = ax.text(0.5, 0.95, '', transform=ax.transAxes,horizontalalignment='center');
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.);

    def init():
        line.set_data([], []);
        maxline.set_data([], []);
        minline.set_data([], []);
        dot.set_data([], []);
        time_text.set_text('');

        return line, time_text, dot, maxline, minline

    def animate(i):
        thisx = xvec;
        thisy = umat[:,i];

        maxx = [0,L];
        maxy = [u_max[i],u_max[i]];

        minx = [0,L];
        miny = [u_min[i],u_min[i]];

        line.set_data(thisx, thisy);
        maxline.set_data(maxx, maxy);
        minline.set_data(minx, miny);
        dot.set_data([0],[umat[0,i]]);
        time_text.set_text(time_template % (tvec[i]));
        return line, time_text, dot, maxline, minline
    
    skip = 10
    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(tvec),1+skip),
                                  interval=50, blit=True, init_func=init,repeat=False);
    
    plt.close(fig);
    
    ani_jshtml = ani.to_jshtml(fps=5)
    ani_html = HTML(ani_jshtml)
    clear_output()
    display(ani_html)
    display(Markdown('# You can slow down the animation (if you want) by using the - button.'))
    
    pass