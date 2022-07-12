import os
from tkinter import *
from threading import Thread
from PIL import ImageTk, Image
import tkinter as tk


window=Tk()

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

window.title("PHYSICS SIMULATION LIST")
window.geometry("1920x1080")
window.resizable(True, True)

if screen_width>1920:
    Wvar= screen_width/1920
    Hvar= screen_height/1080
else:
    Wvar= 1920/screen_width
    Hvar= 1080/screen_height

wsize=512
hsize=512

nsize=512*Wvar
msize=int(nsize)

image1 = Image.open('Assets\Doublep.png')
new_image = image1.resize((msize, msize), Image.ANTIALIAS)
new_image.save('Assets\Doublep_n.png')
imcu1=PhotoImage(file=("Assets\Doublep_n.png"))

image2 = Image.open('Assets\young2-.png')
new_image = image2.resize((msize, msize), Image.ANTIALIAS)
new_image.save('Assets\young_n.png')
imcu2=PhotoImage(file=("Assets\young_n.png"))

image3 = Image.open('Assets\electofield.png')
new_image = image3.resize((msize, msize), Image.ANTIALIAS)
new_image.save('Assets\electofield_n.png')
imcu3=PhotoImage(file=("Assets\electofield_n.png"))

image4 = Image.open('Assets\WAVE.png')
new_image = image4.resize((msize, msize), Image.ANTIALIAS)
new_image.save('Assets\WAVE_n.png')
imcu4=PhotoImage(file=("Assets\WAVE_n.png"))

image5 = Image.open('Assets\dopler.png')
new_image = image5.resize((msize, msize), Image.ANTIALIAS)
new_image.save('Assets\dopler_n.png')
imcu5=PhotoImage(file=("Assets\dopler_n.png"))


def run1_threaded():
        Thread(target=run1()).start()

def run1():

    import matplotlib
    matplotlib.use('TkAgg') 
    import matplotlib.pyplot as plt 
    import scipy as sp
    import matplotlib.animation as animation

    def runSim1_threaded(c11, c12, c13):
        Thread(target=runSim1(c11, c12, c13)).start()
        
    def runSim1(c11, c12, c13):
        pball1 = pball(theta1=sp.pi, theta2=sp.pi - 0.01, dt=0.01, p1=c11, p2=c12, length =c13)
        anime = Anime(pendulum=pball1, draw_trace=True, TEMP=c13)
        anime.animate()
        plt.show()
 
    class pball:
        def __init__(self, theta1, theta2, dt, p1, p2, length):
            self.theta1 = theta1
            self.theta2 = theta2
            self.p1 = p1
            self.p2 = p2
            self.dt = dt
          
            self.g = 9.81
            self.length = length
          
            self.trajectory = [self.polar_to_cartesian()]
  
        def polar_to_cartesian(self):
            x1 =  self.length * sp.sin(self.theta1)        
            y1 = -self.length * sp.cos(self.theta1)
          
            x2 = x1 + self.length * sp.sin(self.theta2)
            y2 = y1 - self.length * sp.cos(self.theta2)
         
            print(self.theta1, self.theta2)
            return sp.array([[0.0, 0.0], [x1, y1], [x2, y2]])
      
        def evolve(self):
            theta1 = self.theta1
            theta2 = self.theta2
            p1 = self.p1
            p2 = self.p2
            g = self.g
            l = self.length
         
            expr1 = sp.cos(theta1 - theta2)
            expr2 = sp.sin(theta1 - theta2)
            expr3 = (1 + expr2**2)
            expr4 = p1 * p2 * expr2 / expr3
            expr5 = (p1**2 + 2 * p2**2 - p1 * p2 * expr1) \
            * sp.sin(2 * (theta1 - theta2)) / 2 / expr3**2
            expr6 = expr4 - expr5
         
            self.theta1 += self.dt * (p1 - p2 * expr1) / expr3
            self.theta2 += self.dt * (2 * p2 - p1 * expr1) / expr3
            self.p1 += self.dt * (-2 * g * l * sp.sin(theta1) - expr6)
            self.p2 += self.dt * (    -g * l * sp.sin(theta2) + expr6)
         
            new_position = self.polar_to_cartesian()
            self.trajectory.append(new_position)
            print(new_position)
            return new_position
 
 
    class Anime:
        def __init__(self, TEMP , pendulum, draw_trace=False ):
            self.pendulum = pendulum
            self.draw_trace = draw_trace
            self.time = 0.0
            self.TEMP = TEMP
  
   
            self.fig, self.ax = plt.subplots()
            self.ax.set_ylim(-1.0-TEMP*2, 1.0+TEMP*2)
            self.ax.set_xlim(-1.0-TEMP*2, 1.0+TEMP*2)
  
            self.time_text = self.ax.text(0.05, 0.95, '', 
                horizontalalignment='left', 
                verticalalignment='top', 
                transform=self.ax.transAxes)
  
        
            self.line, = self.ax.plot(
                self.pendulum.trajectory[-1][:, 0], 
                self.pendulum.trajectory[-1][:, 1], 
                marker='o')
          
   
            if self.draw_trace:
                self.trace, = self.ax.plot(
                    [a[2, 0] for a in self.pendulum.trajectory],
                    [a[2, 1] for a in self.pendulum.trajectory])
     
        def advance_time_step(self):
            while True:
                self.time += self.pendulum.dt
                yield self.pendulum.evolve()
             
        def update(self, data):
            self.time_text.set_text('Elapsed time: {:6.2f} s'.format(self.time))
         
            self.line.set_ydata(data[:, 1])
            self.line.set_xdata(data[:, 0])
         
            if self.draw_trace:
                self.trace.set_xdata([a[2, 0] for a in self.pendulum.trajectory])
                self.trace.set_ydata([a[2, 1] for a in self.pendulum.trajectory])
            return self.line,
     
        def animate(self):
            self.animation = animation.FuncAnimation(self.fig, self.update,
                             self.advance_time_step, interval=25, blit=False)


    def quit1():
    	top1.destroy()
    	


    def getSimVals1():

        c11 = float(lam.get())
        c12 = float(Bar.get())
        c13 = float(Aero.get())
    
        runSim1_threaded(c11, c12, c13)

    def mkMain1():
        
        global lam,Bar,Aero
        
        Quit1 = Button(top1,text = " Quit ",activeforeground='red',activebackground='gray', command = quit1).grid(row = 9, column = 2) 
        run = Button(top1,text = " Run ",activeforeground='red',activebackground='gray', command = getSimVals1).grid(row = 9, column = 3) 
        Label(top1,text = 'Enter the mass constant of the first pendulum').grid(row = 1,column=0) 
        lam = Entry(top1,width=6)
        lam.insert(10,'0') 
        lam.grid(row = 1,column=1)

        Label(top1,text = 'Enter the mass constant for second pendulum').grid(row = 2,column=0) 
        Bar = Entry(top1,width=6) 
        Bar.insert(10,'0')
        Bar.grid(row = 2,column=1)
        Label(top1,text = 'Enter the length of the pendulum rod ').grid(row = 3,column=0) 
        Aero = Entry(top1,width=6) 
        Aero.insert(10,'1.0')
        Aero.grid(row = 3,column=1)
        Label(top1,text = '       ').grid(row = 8,column=2)

    global top1

    top1 = Tk()
    top1.title("Double Pendulum Simulation")
    mkMain1() 
    top1.mainloop()	 
 
btn = Button(window, text="Double Pendulum Simulation", image = imcu1 ,command=run1_threaded)
btn.place(x=0, y=0)

def run2_threaded():
        Thread(target=run2()).start()

def run2():

    from numpy import pi, linspace, sin, cos
    import matplotlib.pyplot as plt

    def runSim2_threaded(Lambbda,br,ar,Dr,er):
        Thread(target=runSim2(Lambbda,br,ar,Dr,er)).start()

    def runSim2(Lambbda,br,ar,Dr,er):
        lamda = Lambbda*1.E-9
        b = br*1.E-3
        a = ar*1.E-3
        D = Dr
        e = er*1.E-2
        X_Mmax = e/2. ; X_Mmin = -e/2
        N = 1000
        X = linspace(X_Mmax, X_Mmin, N)
        B = (pi*b*X)/(lamda*D)
        A = (pi*a*X)/(lamda*D)
        I = 0.5*(sin(B)/B)**2 * (1+cos(2*A))
        Envelop = (sin(B)/B)**2
        fig = plt.figure(figsize=(8,5))
        fig.suptitle('Young Double Slit Diffraction', fontsize=14, fontweight='bold')
        
        axis = fig.add_subplot(111)
        axis.grid(True)
        axis.plot(X, I, '-k', linewidth=3)
        axis.plot(X, Envelop, '--k', linewidth=2, alpha=0.8)
        axis.set_xlim(X_Mmin, X_Mmax)
        axis.set_xlabel(r'$X/(m)$', fontsize=14, fontweight='bold')
        axis.set_ylabel(r'$I(X, Y)/I_0$', fontsize=14, fontweight='bold')
        axis.set_title(r"$wavelength = %.1e m, b = %.1e m, a = %.1e m, D = %.0e m$"%(lamda,b,a,D))
        plt.show()



    def quit2():
        top2.destroy()
    
    def getSimVals2():
        Lambbda = float(lam.get()) 
        br = float(Bar.get())
        ar = float(Aero.get())
        Dr = float(dis.get())
        er = float(Ee.get())
        
        runSim2_threaded(Lambbda,br,ar,Dr,er)
        
    def mkMain2():
        
        global lam,Bar,Aero,dis,Ee
        
        Quit2 = Button(top2,text = " Quit ",activeforeground='red',activebackground='gray', command = quit2).grid(row = 12, column = 2) 
        runn2 = Button(top2,text = " Run ",activeforeground='red',activebackground='gray', command = getSimVals2).grid(row = 12, column = 3) 
        
        Label(top2,text = 'Wave Length(nm)').grid(row = 1,column=0) 
        lam = Entry(top2,width=6)
        lam.insert(10,'0') 
        lam.grid(row = 1,column=1)
        
        Label(top2,text = 'Slit width(b)(mm)').grid(row = 2,column=0) 
        Bar = Entry(top2,width=6) 
        Bar.insert(10,'0')
        Bar.grid(row = 2,column=1)
        
        Label(top2,text = 'Inter Slit distance(a)(mm) ').grid(row = 3,column=0) 
        Aero = Entry(top2,width=6) 
        Aero.insert(10,'0')
        Aero.grid(row = 3,column=1)

        Label(top2,text = 'Distance between screen and slit(D)(m) ').grid(row = 4,column=0) 
        dis = Entry(top2,width=6) 
        dis.insert(10,'0')
        dis.grid(row = 4,column=1)
        Label(top2,text = '  ').grid(row = 7,column=0) 
        Label(top2,text = 'Size of the screen(e)(cm) ').grid(row = 8,column=0) 
        Ee = Entry(top2,width=6) 
        Ee.insert(10,'0')
        Ee.grid(row = 8 ,column=1)

    global top2
    top2 = Tk()
    top2.title(" Young's double Slit Simulation")
    mkMain2()
    top2.mainloop()	


btn1 = Button(window, text=" Young's double Slit Simulation", image = imcu2 ,command=run2_threaded)
btn1.place(x=1390*Wvar, y=0)



def run3_threaded():
        Thread(target=run3()).start()

def run3():
    
    import numpy as np
    from matplotlib import pyplot as plt
    plt.style.use('bmh')

    def runSim3_threaded(c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43,c51,c52,c53):
        Thread(target=runSim3(c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43,c51,c52,c53)).start()
    
    def runSim3(c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43,c51,c52,c53):
        
        plt.figure(figsize=(8, 8))

        C = [(c11,c12,c13), (c21,c22,c23), (c31,c32,c33), (c41,c42,c43), (c51,c52,c53)] 
        [xmin, xmax, ymin, ymax] = [-10, 10, -10, 10]
        k = 8.99*10**9  
        
        for i in range(0, len(C)):
            if C[i][2] > 0:
                color = 'bo'
            elif C[i][2] < 0:
                color = 'ro' 
            else:
                color = 'wo'
            plt.plot(C[i][0], C[i][1], color)
            plt.axis([xmin, xmax, ymin, ymax])
        n = 200j
        Y, X = np.mgrid[xmin:xmax:n, ymin:ymax:n]  
        Ex, Ey = np.array(X*0), np.array(Y*0)
        
        for i in range(0, len(C)):
            R = np.sqrt((X-C[i][0])**2 + (Y-C[i][1])**2)
            Ex = Ex + k*C[i][2]/R**2*(X-C[i][0])/R
            Ey = Ey + k*C[i][2]/R**2*(Y-C[i][1])/R
       
        
        plt.figure(figsize=(8, 8))
        V = 0*X
            
        for i in range(0, len(C)):
            R = np.sqrt((X-C[i][0])**2 + (Y-C[i][1])**2)
            V = V + k*C[i][2]/R
        
        Ey, Ex = np.gradient(-V)
        equip_surf = np.linspace(np.min(V)*0.05, np.max(V)*0.05, 20)
        
        plt.streamplot(X, Y, Ex, Ey, color='k', density=1, arrowstyle='simple')
        contour_surf = plt.contour(X, Y, V, equip_surf)
        plt.colorbar(contour_surf, shrink=0.8, extend='both')
        plt.xlabel('x, [m]')
        plt.ylabel('y, [m]')
        plt.show()
        
    def quit3():
            top3.destroy()
        
        
    def getSimVals3():
            
            dt = 0.005 
            
            c11 = float(p11.get())
            c12 = float(p12.get())
            c13 = float(p13.get())
            c21 = float(p21.get())
            c22 = float(p22.get())
            c23 = float(p23.get())
            c31 = float(p31.get())
            c32 = float(p32.get())
            c33 = float(p33.get())
            c41 = float(p41.get())
            c42 = float(p42.get())
            c43 = float(p43.get())
            c51 = float(p51.get())
            c52 = float(p52.get())
            c53 = float(p53.get())
            
            runSim3_threaded(c11,c12,c13,c21,c22,c23,c31,c32,c33,c41,c42,c43,c51,c52,c53)
            
    def mkMain3():
        
        global p11,p12,p13,p21,p22,p23,p31,p32,p33,p41,p42,p43,p51,p52,p53
        Quit3 = Button(top3,text = " Quit ",activeforeground='red',activebackground='gray', command = quit3).grid(row = 9, column = 2) 
        runn3 = Button(top3,text = " Run ",activeforeground='red',activebackground='gray', command = getSimVals3).grid(row = 9, column = 3) 
        Label(top3,text = ' coordinate (x,y) and charge of particles (z) ').grid(row = 1,column=0)
        p11 = Entry(top3,width=6)
        p11.insert(10,'0') 
        p11.grid(row = 1,column=1)
        p12 = Entry(top3,width=6)
        p12.insert(10,'0')  
        p12.grid(row = 1,column=2)
        p13 = Entry(top3,width=6)
        p13.insert(10,'0') 
        p13.grid(row = 1,column=3)
        p21 = Entry(top3,width=6)
        p21.insert(10,'0') 
        p21.grid(row = 2,column=1)
        p22 = Entry(top3,width=6)
        p22.insert(10,'0') 
        p22.grid(row = 2,column=2)
        p23 = Entry(top3,width=6)
        p23.insert(10,'0') 
        p23.grid(row = 2,column=3)
        p31 = Entry(top3,width=6)
        p31.insert(10,'0') 
        p31.grid(row = 3,column=1)
        p32 = Entry(top3,width=6)
        p32.insert(10,'0') 
        p32.grid(row = 3,column=2)
        p33 = Entry(top3,width=6)
        p33.insert(10,'0') 
        p33.grid(row = 3,column=3)
        p41 = Entry(top3,width=6)
        p41.insert(10,'0') 
        p41.grid(row = 4,column=1)
        p42 = Entry(top3,width=6)
        p42.insert(10,'0') 
        p42.grid(row = 4,column=2)
        p43 = Entry(top3,width=6)
        p43.insert(10,'0') 
        p43.grid(row = 4,column=3)
        p51 = Entry(top3,width=6)
        p51.insert(10,'0') 
        p51.grid(row = 5,column=1)
        p52 = Entry(top3,width=6)
        p52.insert(10,'0') 
        p52.grid(row = 5,column=2)
        p53 = Entry(top3,width=6)
        p53.insert(10,'0') 
        p53.grid(row = 5,column=3)
        
        Label(top3,text = '       ').grid(row = 8,column=2)
        
    global top3
    top3 = Tk()
    top3.title("Electric fields simulation")
    mkMain3() 
    top3.mainloop()



btn2 = Button(window, text=" Electric fields simulation", image = imcu3 ,command=run3_threaded)
btn2.place(x=700*Wvar, y=280*Hvar)


def run6_threaded():
        Thread(target=run6()).start()

def run6():

    from math import sqrt,sin,pi
    from numpy import empty
    from pylab import imshow,gray,show

    def runSim4_threaded(Lambbda,br):
        Thread(target=runSim4(Lambbda,br)).start()
    
    
    def runSim4(Lambbda,br):
        wavelength = Lambbda
        k = 2*pi/wavelength
        xi0 = 1.0
        separation = br      
        side = 100.0           
        points = 1000           
        spacing = side/points  
        x1 = side/2 + separation/2
        y1 = side/2
        x2 = side/2 - separation/2
        y2 = side/2
        xi = empty([points,points],float)
        
        for i in range(points):
            y = spacing*i
            for j in range(points):
                x = spacing*j
                r1 = sqrt((x-x1)**2+(y-y1)**2)
                r2 = sqrt((x-x2)**2+(y-y2)**2)
                xi[i,j] = xi0*sin(k*r1) + xi0*sin(k*r2)
        imshow(xi,origin="lower",extent=[0,side,0,side])
        gray()
        show()

    def quit4():
        top4.destroy()
       
    def getSimVals4():
        
        Lambbda = float(lam.get()) 
        br = float(Bar.get())
        
        runSim4_threaded(Lambbda,br)
        
    def mkMain4():
        
        global lam,Bar
        Quit4 = Button(top4,text = " Quit ",activeforeground='red',activebackground='gray', command = quit4).grid(row = 12, column = 2) 
        runn4 = Button(top4,text = " Run ",activeforeground='red',activebackground='gray', command = getSimVals4).grid(row = 12, column = 3) 
        
        Label(top4,text = 'Wave Length(mm)').grid(row = 1,column=0) 
        lam = Entry(top4,width=6)
        lam.insert(10,'0') 
        lam.grid(row = 1,column=1)
        
        Label(top4,text = 'Wave separation in cm').grid(row = 2,column=0) 
        Bar = Entry(top4,width=6) 
        Bar.insert(10,'0')
        Bar.grid(row = 2,column=1)
        
    global top4
    top4 = Tk()
    top4.title("Wave Pattern Simulation")
    mkMain4()
    top4.mainloop()	


btn3 = Button(window, text=" Wave Pattern Simulation", image = imcu4 ,command=run6_threaded)
btn3.place(x=0, y=585*Hvar)



def run7_threaded():
        Thread(target=run7()).start()

def run7():
    
    import numpy as np
    import matplotlib.pyplot as plt

    def runSim5_threaded(dt,vw,vs,psi,vo,poi):
        Thread(target=runSim5(dt,vw,vs,psi,vo,poi)).start()
    
    def runSim5(dt,vw,vs,psi,vo,poi):
        
        plt.ion()

        fig = plt.figure(figsize=(6,6))
        ax3 = fig.add_subplot(111) 
        ax1 = fig.add_subplot(421) 
        ax2 = fig.add_subplot(422) 
        xsz = [0,5] 
        ysz = [-2,4] 
        
        circs = [] 
        
        xsc = psi[0] 
        ysc = psi[1]
        xoc = poi[0]
        yoc = poi[1]
        
        ax1.set_xlim([0,dt*4]) 
        ax1.set_ylim([-1,1]) 
        ax1.set_title('Source')
        
        time = np.linspace(0,dt*4,100) 
        ysource = [np.sin(np.pi*(1/dt)*k) for k in time]
        ax1.plot(time,ysource)
        
        wcountKeep = []
        listenRange = 5.0
        ct = 0 
        
        while True:
            xsc+=vs[0]*dt
            ysc+=vs[1]*dt
            xoc+=vo[0]*dt
            yoc+=vo[1]*dt
            
            ax3.cla()
            ax2.cla()
            
            ax3.set_xlim([xsz[0],xsz[1]]) 
            ax3.set_ylim([ysz[0],ysz[1]]) 
            ax2.set_xlim([0,dt*4]) 
            ax2.set_ylim([-1,1]) 
            ax2.set_title('Observer')
            
            ax3.plot(xsc,ysc,'ro')
            ax3.plot(xoc,yoc,'bo')
            
            if ct%2 == 0:
                circs.append(circ(xsc,ysc))
                
            if len(circs) > 100: 
                del circs[0]
                
            
            wcount = 0
            
            for cir in circs: 
                cir.updateCirc(cir.r+dt*vw)
                ax3.add_artist(cir.circ) 
                
                xtemp = cir.x 
                ytemp = cir.y 
                rtemp = cir.r 
                centDisp = [1.0*xtemp-1.0*xoc,1.0*ytemp-1.0*yoc] 
                centDist = np.sqrt((centDisp[0]**2)+(centDisp[1]**2)) 
                vorptw = -1*((vo[0]*centDisp[0] + vo[1]*centDisp[1])/centDist)
                
                if abs(centDist-rtemp) < listenRange*dt*(vw-vorptw) : 
                    wcount+=1
                    
            wcountKeep.append(wcount) 
            y = [np.sin(((np.average(wcountKeep)/listenRange)*(np.pi/dt)*k)) for k in time]
            ax2.plot(time,y)
            
            if len(wcountKeep)>1: 
                del wcountKeep[0]
                
            fig.canvas.draw()
            fig.canvas.flush_events()
            ct+=1

    class circ: 

        def __init__(self,x,y):
            self.r = 0 
            self.x = x
            self.y = y
            self.circ = plt.Circle((x, y), self.r,facecolor='none',edgecolor='k')
            
        def updateCirc(self,r):
            self.r = r
            self.circ = plt.Circle((self.x, self.y), self.r,facecolor='none',edgecolor='k')
            
    def getSimVals5():
        
        dt = 0.005 
        vw = float(waveSpeedF.get()) 
        vs = [float(vsxF.get()),float(vsyF.get())] 
        psi = [float(psxF.get()),float(psyF.get())] 
        vo = [float(voxF.get()),float(voyF.get())]
        poi = [float(poxF.get()),float(poyF.get())]
        runSim5_threaded(dt,vw,vs,psi,vo,poi)
        
    def quit5():
        top5.destroy()
    
    def mkMain5():
        
        global waveSpeedF,vsxF,vsyF,psxF,psyF,voxF,voyF,poxF,poyF
        
        Quit5 = Button(top5,text = " Quit ",activeforeground='red',activebackground='gray', command = quit5).grid(row = 9, column = 2) 
        runn5 = Button(top5,text = " Run ",activeforeground='red',activebackground='gray', command = getSimVals5).grid(row = 9, column = 3) 
        
        Label(top5,text = 'Wave speed: ').grid(row = 1,column=0) 
        waveSpeedF = Entry(top5,width=6)
        waveSpeedF.insert(10,'20.0') 
        waveSpeedF.grid(row = 1,column=1)
        
        Label(top5,text = '         ').grid(row = 2,column=2)
        Label(top5,text = 'Source velocity: ').grid(row = 3,column=0) 
        vsxF = Entry(top5,width=6) 
        vsxF.insert(10,'5.0')
        vsxF.grid(row = 3,column=1)
        Label(top5,text = ' , ').grid(row = 3,column=2)
        vsyF = Entry(top5,width=6) 
        vsyF.insert(10,'0.0')
        vsyF.grid(row = 3,column=3)
        Label(top5,text = 'Source initial position: ').grid(row = 4,column=0) 
        psxF = Entry(top5,width=6) 
        psxF.insert(10,'0.0')
        psxF.grid(row = 4,column=1)
        Label(top5,text = ' , ').grid(row = 4,column=2)
        psyF = Entry(top5,width=6) 
        psyF.insert(10,'0.0')
        psyF.grid(row = 4,column=3)
        Label(top5,text = '         ').grid(row = 5,column=2)
        Label(top5,text = 'Observer velocity: ').grid(row = 6,column=0) 
        voxF = Entry(top5,width=6) 
        voxF.insert(10,'0.0')
        voxF.grid(row = 6,column=1)
        Label(top5,text = ' , ').grid(row = 6,column=2)
        voyF = Entry(top5,width=6) 
        voyF.insert(10,'0.0')
        voyF.grid(row = 6,column=3)
        Label(top5,text = 'Observer initial position: ').grid(row = 7,column=0)
        poxF = Entry(top5,width=6) 
        poxF.insert(10,'2.0')
        poxF.grid(row = 7,column=1)
        Label(top5,text = ' , ').grid(row = 7,column=2)
        poyF = Entry(top5,width=6) 
        poyF.insert(10,'-1.0')
        poyF.grid(row = 7,column=3)
        Label(top5,text = '       ').grid(row = 8,column=2)
        
    global top5
    top5 = Tk()
    mkMain5() 
    top5.mainloop()
    

btn4 = Button(window, text=" Doppler effect simulation", image = imcu5 ,command=run7_threaded)
btn4.place(x=1390*Wvar, y=585*Hvar)


window.mainloop()
