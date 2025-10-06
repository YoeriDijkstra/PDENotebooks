import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Markdown, clear_output, HTML
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import warnings
# warnings.filterwarnings('ignore',message='overflow encountered in exp')
# warnings.filterwarnings('ignore',message='overflow encountered in multiply')
# warnings.filterwarnings('ignore',message='invalid value encountered in multiply')

def Heat_equation_3d(a,phi,eta,f,t_end,L,t,x):
    
    anew = a/L
    
    display(Markdown('# The figure is being rendered, please be patient.'))

    # symbols, functions and accuracy initialisation
    n = sp.symbols('n')
    g = sp.Function('g')(x)
    h = sp.Function('h')(x,t)
    v = sp.Function('v')(x)
    V = sp.Function('V')(x,t)
    N = 10

    # find function V that satisfies nonhomogeneous time-dependent BC's
    ode_v = sp.diff(v,x,x)
    sol = sp.dsolve(ode_v)
    constants = sp.solve([sol.rhs.subs(x,0)-phi,sol.rhs.subs(x,1)-eta],sp.symbols('C1 C2'))
    V = sol.rhs.subs(constants)

    # define new initial condition and right hand side
    g = f-V.subs(t,0)
    h = -V.diff(t)+anew**2*V.diff(x).diff(x)

    # find the Fourier sine expansion of h(x,t) with respect to x
    hn = sp.Function('hn')(t,n)
    hn = sp.integrate(h*sp.sin(n*sp.pi*x),(x,0,1))*2

    # find the Fourier sine expansion of g(x) with respect to x
    gn = sp.Function('gn')(n)
    gn = sp.integrate(g*sp.sin(n*sp.pi*x),(x,0,1))*2

    # find the Fourier sine expansion of w(t,x) with respect to x and construct w
    s = sp.symbols('s')
    Ln = sp.Function('Ln')(n)
    Ln = (anew*n*sp.pi)**2
    w = sp.Function('w')(x,t)
    w = 0
    for nn in np.arange(1,N+1):
        Lnn = Ln.subs(n,nn)
        hnn = hn.subs(n,nn)
        gnn = gn.subs(n,nn)
        wnn = sp.integrate(hnn.subs(t,s)*sp.exp(Lnn*s),(s,0,t))
        wnn = gnn+wnn
        wnn = wnn*sp.exp(-Lnn*t)
        w = w + wnn*sp.sin(nn*sp.pi*x)

    # make finally u(x,t)
    u = sp.Function('u')(x,t)
    u = w + V

    # evaluate u(x,t) at several x and t and plot
    u_lam = sp.lambdify((x,t),u,modules=['numpy'])
    t_vals = np.linspace(0,t_end,int(t_end)*100+1)
    x_vals = np.linspace(0,1,201)
    tv, xv = np.meshgrid(t_vals, x_vals)
    uv = u_lam(xv,tv)
    if isinstance(f,int):
        uv[:,0] = f
    else:
        fnum = sp.lambdify(x,f,"numpy")
        uv[:,0] = fnum(x_vals)
    uv = np.where(np.isinf(uv),np.zeros_like(uv),uv)
    uv = np.where(np.isnan(uv),np.zeros_like(uv),uv)
    
    clear_output()

    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(L*xv, tv, uv, 
                           linewidth=2,antialiased=False,cmap=cm.jet)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$u(x,t)$')
    
    ax.set_title('$a^2='+sp.latex(a**2)+',\phi(t)='+sp.latex(phi)+',\eta='+str(eta)+',f(x)='+sp.latex(f)+'$')
    plt.show()
    
    pass
    
def Heat_equation_anim(a,phi,eta,f,t_end,t,x):

    display(Markdown('# The animation is being rendered, please be patient.'))

    # symbols, functions and accuracy initialisation
    n = sp.symbols('n')
    g = sp.Function('g')(x)
    h = sp.Function('h')(x,t)
    v = sp.Function('v')(x)
    V = sp.Function('V')(x,t)
    N = 10

    # find function V that satisfies nonhomogeneous time-dependent BC's
    ode_v = sp.diff(v,x,x)
    sol = sp.dsolve(ode_v)
    constants = sp.solve([sol.rhs.subs(x,0)-phi,sol.rhs.subs(x,1)-eta],sp.symbols('C1 C2'))
    V = sol.rhs.subs(constants)

    # define new initial condition and right hand side
    g = f-V.subs(t,0)
    h = -V.diff(t)+a**2*V.diff(x).diff(x)

    # find the Fourier sine expansion of h(x,t) with respect to x
    hn = sp.Function('hn')(t,n)
    hn = sp.integrate(h*sp.sin(n*sp.pi*x),(x,0,1))*2

    # find the Fourier sine expansion of g(x) with respect to x
    gn = sp.Function('gn')(n)
    gn = sp.integrate(g*sp.sin(n*sp.pi*x),(x,0,1))*2

    # find the Fourier sine expansion of w(t,x) with respect to x and construct w
    s = sp.symbols('s')
    Ln = sp.Function('Ln')(n)
    Ln = (a*n*sp.pi)**2
    w = sp.Function('w')(x,t)
    w = 0
    for nn in np.arange(1,N+1):
        Lnn = Ln.subs(n,nn)
        hnn = hn.subs(n,nn)
        gnn = gn.subs(n,nn)
        wnn = sp.integrate(hnn.subs(t,s)*sp.exp(Lnn*s),(s,0,t))
        wnn = gnn+wnn
        wnn = wnn*sp.exp(-Lnn*t)
        w = w + wnn*sp.sin(nn*sp.pi*x)

    # make finally u(x,t)
    u = sp.Function('u')(x,t)
    u = w + V

    # evaluate u(x,t) at several x and t and plot
    u_lam = sp.lambdify((x,t),u,modules=['numpy'])
    t_vals = np.linspace(0,t_end,int(t_end)*100+1)
    x_vals = np.linspace(0,1,201)
    tv, xv = np.meshgrid(t_vals, x_vals)
    uv = u_lam(xv,tv)
    if isinstance(f,int):
        uv[:,0] = f
    else:
        fnum = sp.lambdify(x,f,"numpy")
        uv[:,0] = fnum(x_vals)
    uv = np.where(np.isinf(uv),np.zeros_like(uv),uv)
    uv = np.where(np.isnan(uv),np.zeros_like(uv),uv)
    
    u_min = np.min(np.min(uv))
    u_max = np.max(np.max(uv))
    
    fig = plt.figure(2);
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 1), ylim=(u_min,u_max));
    ax.grid();
    ax.set_ylabel('$u$');
    ax.set_xlabel('$x$');
    
    line, = ax.plot([], [], '-', lw=2,label='$u(x,t)$');
    dot, = ax.plot([], [], 'o', lw=2,label='$\\phi(t)$');
    time_template = '$t$ = %1.3fs';
    time_text = ax.text(0.5, 0.95, '', transform=ax.transAxes,horizontalalignment='center');
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.);
    
    def init():
        line.set_data([], []);
        dot.set_data([], []);
        time_text.set_text('');

        return line, time_text, dot
        
    def animate(i):
        thisx = x_vals;
        thisy = uv[:,i];

        line.set_data(thisx, thisy);
        dot.set_data([0],[uv[0,i]]);
        time_text.set_text(time_template % (t_vals[i]));
        return line, time_text, dot
        
    skip = 0
    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t_vals),1+skip),
                                  interval=1, blit=True, init_func=init,repeat=False);
    plt.close(fig);
    
    ani_jshtml = ani.to_jshtml(fps=15)
    ani_html = HTML(ani_jshtml)
    clear_output()
    display(ani_html)
    display(Markdown('# You can slow down the animation (if you want) by using the - button.'))
    

    pass
    
def Heat_equation(a,phi,eta,f,t_end,L,t,x):

    anew = a/L
    
    display(Markdown('# The animation is being rendered, please be patient.'))

    # symbols, functions and accuracy initialisation
    n = sp.symbols('n')
    g = sp.Function('g')(x)
    h = sp.Function('h')(x,t)
    v = sp.Function('v')(x)
    V = sp.Function('V')(x,t)
    N = 10

    # find function V that satisfies nonhomogeneous time-dependent BC's
    ode_v = sp.diff(v,x,x)
    sol = sp.dsolve(ode_v)
    constants = sp.solve([sol.rhs.subs(x,0)-phi,sol.rhs.subs(x,1)-eta],sp.symbols('C1 C2'))
    V = sol.rhs.subs(constants)

    # define new initial condition and right hand side
    g = f-V.subs(t,0)
    h = -V.diff(t)+anew**2*V.diff(x).diff(x)

    # find the Fourier sine expansion of h(x,t) with respect to x
    hn = sp.Function('hn')(t,n)
    hn = sp.integrate(h*sp.sin(n*sp.pi*x),(x,0,1))*2

    # find the Fourier sine expansion of g(x) with respect to x
    gn = sp.Function('gn')(n)
    gn = sp.integrate(g*sp.sin(n*sp.pi*x),(x,0,1))*2

    # find the Fourier sine expansion of w(t,x) with respect to x and construct w
    s = sp.symbols('s')
    Ln = sp.Function('Ln')(n)
    Ln = (anew*n*sp.pi)**2
    w = sp.Function('w')(x,t)
    w = 0
    for nn in np.arange(1,N+1):
        Lnn = Ln.subs(n,nn)
        hnn = hn.subs(n,nn)
        gnn = gn.subs(n,nn)
        wnn = sp.integrate(hnn.subs(t,s)*sp.exp(Lnn*s),(s,0,t))
        wnn = gnn+wnn
        wnn = wnn*sp.exp(-Lnn*t)
        w = w + wnn*sp.sin(nn*sp.pi*x)

    # make finally u(x,t)
    u = sp.Function('u')(x,t)
    u = w + V

    # evaluate u(x,t) at several x and t and plot
    u_lam = sp.lambdify((x,t),u,modules=['numpy'])
    t_vals = np.linspace(0,t_end,int(t_end)*100+1)
    x_vals = np.linspace(0,1,201)
    tv, xv = np.meshgrid(t_vals, x_vals)
    uv = u_lam(xv,tv)
    if isinstance(f,int):
        uv[:,0] = f
    else:
        fnum = sp.lambdify(x,f,"numpy")
        uv[:,0] = fnum(x_vals)
    uv = np.where(np.isinf(uv),np.zeros_like(uv),uv)
    uv = np.where(np.isnan(uv),np.zeros_like(uv),uv)
    
    u_min = np.min(np.min(uv))
    u_max = np.max(np.max(uv))
    
    fig = plt.figure(2);
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, L), ylim=(0,0.1*L));
    ax.grid();
    ax.set_xlabel('$x$');
    ax.set_aspect('equal','box')
    ax.set_yticks([])
    
    # test plot, afterwards, make []
    x_surf = np.vstack((x_vals,x_vals))*L
    y_surf = 0.1*L*np.ones_like(x_surf)
    y_surf[0,:] = 0
    u_surf = np.vstack((uv[:,0],uv[:,0]))
    clev = np.arange(u_min,u_max,0.001)
    surf = plt.imshow(u_surf, vmin = u_min, vmax = u_max, cmap=plt.cm.coolwarm, origin='lower', 
           extent=[0, L, 0, 0.1*L])
    plt.colorbar(label='$u(x,t)$',location='bottom')
    time_template = '$t$ = %1.3fs';
    time_text = ax.text(0.5, 1.05, '', transform=ax.transAxes,horizontalalignment='center');
    
    # u_surf = np.vstack((uv[:,-1],uv[:,-1]))
    # surf.set(data=u_surf)
    
    def init():
        u_surf = np.vstack((uv[:,0],uv[:,0]))
        surf = plt.imshow(u_surf, vmin = u_min, vmax = u_max, cmap=plt.cm.coolwarm, origin='lower', 
           extent=[0, L, 0, 0.1*L])
        time_text.set_text('');

        return surf, time_text
        
    def animate(i):
        u_surf = np.vstack((uv[:,i],uv[:,i]))
        surf.set(data=u_surf)
        time_text.set_text(time_template % (t_vals[i]));
        return surf, time_text
        
    skip = 0
    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(t_vals),1+skip),
                                  interval=1, blit=True, init_func=None,repeat=False);
    plt.close(fig);
    
    ani_jshtml = ani.to_jshtml(fps=15)
    ani_html = HTML(ani_jshtml)
    clear_output()
    display(ani_html)
    display(Markdown('# You can slow down the animation (if you want) by using the - button.'))
    

    pass