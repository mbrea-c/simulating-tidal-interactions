#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Authors: Manuel Brea & Michael Thiel

Notes: - New versions of the code will be uploaded to the Drive folder provided
       - Parameters for the simulation are at the end of the file
       - Comments will be added to every part of the code soon
       - In the current version, output is provided as a sequence of images in
         the same folder the code is located
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy    import constants as const
from astropy.constants.si import G, M_sun, kpc
from astropy    import units     as u
from math import pi
import copy
import time

plt.rc('font', family='serif')

## GLOBAL_CONSTANTS ##
# __Do not modify__ #
DIM    = 3
T_mult = 1e8 * 365.25 * 24 * 3600
M_mult = 1.0e11 * M_sun.value
###################

def v_mag(vector):
    return np.sqrt(np.dot(vector, vector))

class Galaxy:

    def __init__(self, mass, radius=25*kpc.value, rot=(0., 0.), n_rings=6, c_den=2*pi*25*kpc.value / 100, \
                 lim_rings=(0.2, 0.7), v_0=np.zeros((DIM,)), pos_0=np.zeros((DIM,)), spin=1):

        self.n_rings   = n_rings
        self.mass      = mass
        self.dat       = [np.transpose(np.atleast_2d(pos_0)), np.transpose(np.atleast_2d(v_0)), np.zeros((3, 1))]
        self.c_den     = c_den
        self.rot       = rot
        self.lim_rings = lim_rings
        self.spin     = spin
        self.old_pos   = self.dat[0]

        self.plane     = []

        self.radius    = radius

        self.stars = [[], [], []]

    def orbit(self, pos):
        rot   = self.rot
        spin = self.spin

        r      = pos-self.dat[0][:, 0]
        r_magn = np.sqrt(np.dot(r, r))

        v1 = np.array([np.cos(rot[0]), 0., np.sin(rot[0])])
        v2 = np.array([0., np.cos(rot[1]), np.sin(rot[1])])
        vp = np.cross(v1, v2)
        vp = vp / np.sqrt(np.dot(vp, vp))


        v      = np.sqrt((G.value*self.mass)/r_magn)
        v_r    = np.cross((r)/r_magn, vp)


        return spin*v_r*v + self.dat[1][:, 0]

    def generate(self):

        v1 = np.array([np.cos(self.rot[0]), 0., np.sin(self.rot[0])])
        v2 = np.array([0., np.cos(self.rot[1]), np.sin(self.rot[1])])
        vp = np.cross(v1, v2)

        self.plane.append(v1)
        self.plane.append(v2)
        self.plane.append(vp)


        for i in np.linspace(self.lim_rings[0], self.lim_rings[1], self.n_rings):


            radius = i * self.radius
            parts  = int(round((radius*pi*2) / self.c_den))

            alpha  = (2.0*pi) / parts


            vp = vp / np.sqrt(np.dot(vp, vp))




            for i in range(parts):

                pos   = radius*v1*np.cos(alpha*i) + np.cross(vp, v1*radius)*np.sin(alpha*i) + vp*np.dot(vp, v1*radius)*(1.-np.cos(alpha*i))
                pos   = pos + self.dat[0][:, 0]

                self.stars[0].append(pos)
                self.stars[1].append(self.orbit(pos))

        self.stars[0] = np.array(zip(*self.stars[0]))
        self.stars[1] = np.array(zip(*self.stars[1]))
        self.stars[2] = np.zeros(self.stars[0].shape)

        if self.n_rings == 0:
            self.stars[0] = np.zeros((3, 0))
            self.stars[1] = np.zeros((3, 0))
            self.stars[2] = np.zeros((3, 0))

        print self.stars[0].shape




galaxies_l = []

class View:

    def __init__(self, galaxies, debug=False, timestep=0.01*T_mult, \
                 size=(8, 8), length=5*T_mult, sc="k", gc="y", view_mode=0,\
                 view_mrg=25*kpc.value, view_ang=[0., 0.], n_frames=1):

        self.view_mode = view_mode
        self.framerate = n_frames

        self.fig = plt.figure()
        self.fig.set_size_inches(size)
        self.fig.show()

        self.ax3d = self.fig.add_axes([0., 0., 1., 1.], projection="3d")

        self.mrg      = view_mrg

        self.debug    = debug
        self.avg_err  = 0

        self.timestep = timestep
        self.galaxies = galaxies
        self.sc       = sc
        self.gc       = gc
        self.view_ang = view_ang
        self.length   = length
        self.elapsed  = 0.

        self.r_min    = 0.
        self.ecc      = 0.

        self.find_R_min()

        print "e     = %f"     % (self.ecc)
        print "R_min = %f kpc" % (self.r_min / kpc.value)

        print "---- Press enter to simulate ----"
        raw_input()

        print "INSTANTIATING PARTICLES..."

        for i in self.galaxies:

            i.generate()

        print "BEGINNING SIMULATION..."


    def find_R_min(self):
        c_galaxy = self.galaxies[1]
        m_galaxy = self.galaxies[0]

        s_pos = m_galaxy.dat[0][:, 0]
        c_pos = c_galaxy.dat[0][:, 0]

        s_vel = m_galaxy.dat[1][:, 0]
        c_vel = c_galaxy.dat[1][:, 0]

        m1         = m_galaxy.mass
        m2         = c_galaxy.mass
        r          = s_pos - c_pos
        r_mag      = np.sqrt(np.dot(r, r))
        v_rel      = s_vel - c_vel
        m_red      = (m1 * m2) / (m1 + m2)
        cross_temp = np.sqrt(np.dot(np.cross(v_rel, r), np.cross(v_rel, r)))
        L          = cross_temp * m_red

        r0         = (L**2) / (m_red * G.value * m1 * m2)
        U_eff      =  - G.value * (m1 * m2) / r_mag #+ ((L**2) / (2 * m_red * (r_mag**2)))
        K_eff      = 0.5 * m_red * np.sqrt(np.dot(v_rel, v_rel))**2
        E          = U_eff + K_eff
        ecc        = np.sqrt(1 + (2*E*(L**2)) / (m_red*(G.value*m1*m2)**2))

        r_min      = r0 / (1 + ecc)

        self.r_min  = r_min
        self.ecc    = ecc
        self.galaxies[0].radius = r_min
        self.galaxies[1].radius = r_min

    def _update_all(self):

        for glxy in self.galaxies:

            self._v_verlet(glxy.stars)


            self._v_verlet(glxy.dat)

            glxy.old_pos = np.append(glxy.old_pos, glxy.dat[0], axis=1)

        self.elapsed += self.timestep
        print "Elapsed = %f" % (self.elapsed/T_mult)


    def _v_verlet(self, item):

        t = self.timestep

        g_field = np.zeros(item[0].shape)

        item[1] += t*(1.0/2.0)*item[2]
        item[0] += item[1]*t

        for gx in self.galaxies:
            if not item is gx.dat:
                r_magn   = np.sqrt(np.sum((item[0]-gx.dat[0])**2, axis=0))
                g_field += -((G.value*gx.mass)/r_magn**3) * ((item[0]-gx.dat[0]))

        item[2]  = g_field
        item[1] += (1.0/2.0)*item[2]*t

    def begin(self):

        while(not self.__end_condition()):

            self._update_all()

            if  ((self.elapsed/T_mult) % (1.0 / self.framerate)) < 0.95*self.timestep/T_mult:

                a = "c"
                if a != "c":
                    while a != "c":
                        self.plot()
                        self.fig.show()
                        self.fig.canvas.draw()
                        print("User input; c to continue")
                        a = raw_input()
                        if a == "w":
                            self.view_ang[0] += 20
                        if a == "s":
                            self.view_ang[0] -= 20
                        if a == "d":
                            self.view_ang[1] += 20
                        if a == "a":
                            self.view_ang[1] -= 20
                    a = ""
                else:
                    self.plot()


    def plot(self):

        print("PLOTTING...")

        if   self.view_mode==1:
            ctr = self.galaxies[0].dat[0][:, 0]

        elif self.view_mode==2:
            ctr = self.galaxies[1].dat[0][:, 0]

        else:
            ctr = 0.5*(self.galaxies[0].dat[0][:, 0] + self.galaxies[1].dat[0][:, 0])

        mrg = self.mrg


        plot_vec  = False
        plot_pla  = False
        plot_try  = True
        num_rmin  = True

        if self.debug:
            plot_pla  = True
            num_rmin  = True
            plot_try  = True

        self.ax3d.cla()

        self.ax3d.view_init(azim=self.view_ang[0], elev=self.view_ang[1])
        self.ax3d.set_axis_off()
        #self.ax3d.set_facecolor("#183a7a")

        self.ax3d.set_ylabel("y")
        self.ax3d.set_xlabel("x")
        self.ax3d.set_zlabel("z")

        self.ax3d.set_ylim((ctr[1]-mrg, ctr[1]+mrg))
        self.ax3d.set_xlim((ctr[0]-mrg, ctr[0]+mrg))
        self.ax3d.set_zlim((ctr[2]-mrg, ctr[2]+mrg))

        def __draw(self, galaxies, sc=self.sc):

            for gx in self.galaxies:
                stars_pos = gx.stars[0]

                self.ax3d.scatter(*gx.dat[0], marker="o", edgecolor=self.gc, facecolor=self.gc, s=180 * gx.mass/M_mult, depthshade=False)
                self.ax3d.scatter(*stars_pos, marker="o", s=0.02, edgecolor=self.sc, facecolor=self.sc, depthshade=False)

                if plot_try:
                    self.ax3d.plot(*gx.old_pos, ls="--", c="k")


                if plot_pla and gx.n_rings:
                    col = ["r", "b", "g"]
                    lab = ["x", "y", "z"]
                    for ind in xrange(len(gx.plane)):
                        i = gx.plane[ind]
                        v_vec = np.append(gx.dat[0], gx.dat[0] + np.atleast_2d(i).transpose()*0.85*mrg, axis=1)

                        self.ax3d.plot(*v_vec, c=col[ind], ls="-", lw=2)
                        self.ax3d.text(*(v_vec[:, 1]+np.array([0, 5*kpc.value, 0.])), s=lab[ind], size=10, zorder=3, color="k")

                ## PLOTTING VELOCITY VECTORS IF NEEDED
                if plot_vec:
                    col = "k"
                    for i in xrange(gx.stars[0].shape[1]):
                        self.ax3d.plot(*np.append(np.atleast_2d(gx.stars[0][:, i]).transpose(), np.atleast_2d(gx.stars[0][:, i] + 0.1*T_mult*gx.stars[1][:, i]).transpose(), axis=1), c=col)


        __draw(self, self.galaxies)

        fname = self.elapsed / T_mult

        self.fig.canvas.draw()
        self.fig.savefig("Plot%f.png" % fname, dpi=300)
        #self.fig.savefig("NGC5426_%d.eps" % int(fname))

    def __end_condition(self):
        if self.elapsed > self.length:
            self.plot()
            return True
        else:
            return False


######################
# INITIALIZE GALAXIES
######################


##### GALAXY 1 PARAMETERS
#########################
galaxies_l.append(Galaxy(
mass      = 1.0*M_mult,\
# Mass of the center particle. M_mult is set to 10**11 by default
# Referencing the list of all galaxies, don't change
n_rings   = 50,\
# Number of orbiting rings of test particles. Heavily impacts performance
rot       = (0, -(1.0/6.0)*pi), \
# Rotation in radians of the galaxy in global simulation reference system. (ry, rx) where ry is rotation
# around the global Y axis and X is rotation around the global X axis
v_0       = np.array([0., -70000., 0.]), \
# Vector with initial velocity. Must be a numpy array
pos_0     = np.array([0.0e20, 1.0e20, 0.]), \
# Vector with initial position. Must be a numpy array
lim_rings = (0.2, 0.6), \
# Radius of smallest and largest rings, in the form of a fraction of the eventual pericenter distance
c_den     = 2*pi*25*kpc.value / 1050., \
# Constant that defines the circular density of particles (n_particles = 2 * pi * ring_radius / constant)
# It is recommended to only modify the denominator, but feel free to experiment.
# Heavily impacts performance as well.
spin      = 1))
# Spin sense of the orbiting particles. Set to either 1 or 0 for clockwise or anti-clockwise

##### GALAXY 2 PARAMETERS
#########################
galaxies_l.append(Galaxy(
mass      = 1.0*M_mult, \
# Mass of the center particle; M_mult is set to 10^11 times the mass of the sun by default
n_rings   = 50, \
# Number of orbiting rings of test particles. Heavily impacts performance
rot       = (0., -pi/3.0), \
# Rotation in radians of the galaxy in global simulation reference system. (ry, rx) where ry is rotation
v_0       = np.array([0.,  70000., 0.]), \
# Vector with initial velocity. Must be a numpy array
pos_0     = np.array([-14.0e20, -10.0e20, 0.]), \
# Vector with initial position. Must be a numpy array
lim_rings = (0.2, 0.6), \
# Radius of smallest and largest rings, in the form of a fraction of the eventual pericenter distance
c_den     = 2*pi*25*kpc.value / 1050., \
# Constant that defines the circular density of particles (n_particles = 2 * pi * ring_radius / constant)
# It is recommended to only modify the denominator, but feel free to experiment.
# Heavily impacts performance as well.
spin      = 1))
# Spin sense of the orbiting particles. Set to either 1 or 0 for clockwise or anti-clockwise


##### GLOBAL PARAMETERS
#######################
view = View(galaxies_l,\
length    = 10*T_mult, \
# Length of the simulation. T_mult is set to 10^8 years by default
sc        = "k", \
# Color of stars / test particles
gc        = "k", \
# Color of center of mass
debug     = True, \
# Debug mode enables the plotting of spin plane and forces trajectory to be drawn
timestep  = 0.001*T_mult, \
# Physics update timestep. T_mult is set to 10^8 years by default.
# Heavily impacts performance
view_mode = 0, \
# Set to 1 for the figures to be centered on the first galaxy.
# Set to 0 for the figures to be centered on the mid-point between the two galaxies
view_mrg  = 20*kpc.value, \
# Width of the figures.
view_ang  = [115, -7], \
# Angle of the "camera" with respect to global coordinates, in degrees
n_frames  = 1)
# Number of frames to render per T_mult amount of time (10^8 years by default)
#######################


start = time.time()
view.begin()
end   = time.time()
print "Simulation took %f seconds" % (end - start)
