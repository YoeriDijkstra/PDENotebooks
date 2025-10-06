import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, clear_output
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib
# matplotlib.rcParams['animation.embed_limit'] = 2**128

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self,n,m):
        
        self.N = n
        self.M = m
        
        fig = plt.figure(4)
        self.ax1 = fig.add_subplot(1, 1, 1)
        
        self.t = np.linspace(0, 4*np.pi, 401)
        
        self.ax1.set_title('$t$ = %1.2f' % (0))
        self.line1 = Line2D([], [], color='black')
        self.line1e = Line2D(
            [], [], color='black', marker='o', markeredgecolor='black',linestyle='None')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1e)
        self.ax1.set_xlim(-0.1, np.pi+0.1)
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.axis('off')
        
        animation.TimedAnimation.__init__(self, fig, interval=100, blit=True,repeat=False)
        
        plt.close(fig);

    def _draw_frame(self, framedata):
        i = framedata
        
        x = np.linspace(0,np.pi,101)
        self.line1.set_data(x, np.sin(self.N*x)*np.cos(self.N*self.t[i])+1/2*np.sin(self.M*x)*np.cos(self.M*self.t[i]))
        if self.M == 0:
            xnodes = np.linspace(0,np.pi,num=self.N+1)
            ynodes = np.zeros_like(xnodes)
        elif self.M % self.N == 0 or self.N % self.M == 0:
            xnodes = np.linspace(0,np.pi,num=np.min([self.N,self.M])+1)
            ynodes = np.zeros_like(xnodes)
        else:
            xnodes = np.array([0,np.pi])
            ynodes = np.zeros_like(xnodes)
        self.line1e.set_data(xnodes,ynodes)
        
        self.ax1.set_title('$t$ = %1.2f$\pi$' % (self.t[i]/np.pi))

        self._drawn_artists = [self.line1, self.line1e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [self.line1, self.line1e]
        for l in lines:
            l.set_data([], [])

def Standing_waves(n,m):
    
    ani = SubplotAnimation(n,m)
    clear_output()
    display(Markdown('# The animation is being rendered, please be patient.'))
    ani_jshtml = ani.to_jshtml()
    ani_html = HTML(ani_jshtml)
    clear_output()
    display(ani_html)
    display(Markdown('# You can slow down the animation (if you want) by using the - button.'))
    
    pass