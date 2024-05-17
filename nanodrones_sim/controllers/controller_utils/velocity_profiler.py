import numpy as np


class VelocityProfiler:
    def __init__(self,ax_max=3, ax_min=3, ay_max=1):
        # self.a = 5
        self.ax_max = ax_max
        self.ax_min = ax_min

        self.ay_max = ay_max
        self.max_velocity = 10


    def run(self, traj):
        k, v_max = self.get_max_velocity(traj)
        v_max[0] = 0.2
        v = self.get_real_velocity(traj, v_max, k)
        return v

    def get_real_velocity(self, points, v_max, k):
        n = len(v_max)
        ax = np.zeros(n)
        ay = np.zeros(n)

        ### Find min apex
        first_apex_index = None
        min_apex = float('inf')
        n = len(v_max)
        for i in range(n):
            if v_max[i-1] < v_max[i] < v_max[(i+1) % n]:
                if v_max[i] < min_apex:
                    first_apex_index = i
                    min_apex = v_max[i]

        v_real = np.zeros_like(v_max)
        v_real[first_apex_index] = v_max[first_apex_index]

        ### Forward vel calculation
        i = first_apex_index + 1
        while i != first_apex_index:
            ds = np.linalg.norm(points[i] - points[i-1])

            ay[i] = min(self.ay_max, v_real[i - 1] ** 2 * k[i])
            ax[i] = self.ax_max * np.sqrt(1 - (ay[i] / self.ax_max) ** 2) # - self.drag_coeff * v_real[i - 1] ** 2

            calc_vel = v_real[i-1] + (ax[i] / v_real[i-1]) * ds
            v_real[i] = min(v_max[i], calc_vel)

            i = (i+1) % n


        i = first_apex_index - 1
        while i != first_apex_index:
            prec_i = (i+1) % n
            ds = np.linalg.norm(points[prec_i] - points[i])

            ay_b = min(self.ay_max,v_real[prec_i]**2*k[prec_i])
            ax_b = self.ax_min*np.sqrt(1 - (ay_b/self.ax_min)**2 )# + self.drag_coeff * v_real[prec_i]**2

            calc_vel = v_real[prec_i] + (ax_b / v_real[prec_i]) * ds


            if calc_vel < v_real[i]:  # braking
                ax[i] = -ax_b
                ay[i] = ay_b

            v_real[i] = min(v_real[i], calc_vel)

            i = i-1
            if i < 0:
                i = n+i

        return v_real

    def get_max_velocity(self, points):
        curve_points = points[::]

        curvature_coefficients_k = self.get_curvature_coefficients(curve_points)

        v = np.sqrt(self.ay_max/curvature_coefficients_k)
        v[v > self.max_velocity] = self.max_velocity

        return curvature_coefficients_k, v

    def get_curvature_coefficients(self, track):
        n = len(track)

        k = []
        x = track[:,0]
        y = track[:,1]

        dx = x - np.roll(x, 1)
        dy = y - np.roll(y, 1)
        # print("AAA",np.sum(dx == np.nan))
        # print(dx)

        T = np.vstack((dx, dy))

        T_norm = np.linalg.norm(T, axis=0)

        T = T / T_norm

        Tx = T[0]
        Ty = T[1]

        dT_x = Tx - np.roll(Tx,1)
        dT_y = Ty - np.roll(Ty,1)


        ds = np.sqrt( np.power(dx, 2) + np.power(dy, 2) )
        # print(ds)

        dTds_x_norm = dT_x / ds
        dTds_y_norm = dT_y / ds

        k = np.sqrt(np.power(dTds_x_norm,2) + np.power(dTds_y_norm,2))
        # print(k)

        return k
