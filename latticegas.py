
import numpy as np
import os


class LatticeGas:

    def __init__(self, parametrs: dict, obstacle: dict):
        self.nx = parametrs['nx']
        self.ny = parametrs['ny']

        if 'N' in parametrs:
            self.N = parametrs['N']
            self.u_lb, self.Re = self._calculate_params_from_N(self.N)
            print(f"✅ Используется вариант N={self.N}: u_lb={self.u_lb:.4f}, Re={self.Re}")
        else:
            self.u_lb = parametrs['u_lb']
            self.Re = parametrs['Re']
            self.N = None

        self.v = np.array([
            [1, 1], [1, 0], [1, -1],
            [0, 1], [0, 0], [0, -1],
            [-1, 1], [-1, 0], [-1, -1]
        ], dtype=float)

        self.alpha = np.array([
            1/36, 1/9, 1/36,
            1/9, 4/9, 1/9,
            1/36, 1/9, 1/36
        ], dtype=float)

        r = obstacle['r']
        self.nu = (self.u_lb * r) / self.Re
        self.w = min(1.0 / (3.0 * self.nu + 0.5), 1.9)

        self.f_in = np.zeros((9, self.nx, self.ny))
        self.f_out = np.zeros((9, self.nx, self.ny))
        self.rho = np.ones((self.nx, self.ny))
        self.u = np.zeros((self.nx, self.ny, 2))

        self.obstacle = self.add_cylinder(
            obstacle['xc'], obstacle['yc'], obstacle['r'], (self.nx, self.ny)
        )

        self.field_den = []
        self.field_u = []
        self.field_p = []
        self.field_steps = []  

        self._initialize_fields()

    

    @staticmethod
    def _calculate_params_from_N(N):
        u_lb = 0.01 + 2 * (0.1 - 0.01) / (N - 1)
        Re = int(20 + 2 * (1000 - 20) / (N - 1))
        return u_lb, Re

    

    def _initialize_fields(self):
        self.u[:, :, 0] = self.u_lb
        self.rho[:, :] = 1.0
        for i in range(9):
            self.f_in[i] = self.calc_f_eq_i(i, self.u, self.rho)

    

    @staticmethod
    def add_cylinder(xc, yc, r, shape):
        nx, ny = shape
        x, y = np.ogrid[:nx, :ny]
        mask = (x - xc)**2 + (y - yc)**2 <= r**2
        coords = np.where(mask)
        return {'x': coords[0].tolist(), 'y': coords[1].tolist()}

    

    @staticmethod
    def calc_outflow(f_in):
        f_in[6, -1, :] = f_in[6, -2, :]
        f_in[7, -1, :] = f_in[7, -2, :]
        f_in[8, -1, :] = f_in[8, -2, :]

    @staticmethod
    def calc_u(rho, f_in, v):
        ux = np.zeros_like(rho)
        uy = np.zeros_like(rho)
        for i in range(9):
            ux += f_in[i] * v[i, 0]
            uy += f_in[i] * v[i, 1]
        ux /= rho
        uy /= rho
        return np.stack([np.nan_to_num(ux), np.nan_to_num(uy)], axis=-1)

    

    def calc_f_eq_i(self, i, u, rho):
        v_dot_u = self.v[i, 0] * u[:, :, 0] + self.v[i, 1] * u[:, :, 1]
        u_sqr = u[:, :, 0]**2 + u[:, :, 1]**2
        return rho * self.alpha[i] * (1 + 3*v_dot_u + 4.5*v_dot_u**2 - 1.5*u_sqr)

    

    def calc_inflow(self):
        self.u[0, :, 0] = self.u_lb
        rho_known = np.sum(self.f_in[[0,1,2,3,4,5], 0, :], axis=0)
        self.rho[0, :] = np.maximum(rho_known / (1 - self.u[0, :, 0]), 1e-6)
        for i in [6,7,8]:
            self.f_in[i, 0, :] = self.calc_f_eq_i(i, self.u[0:1], self.rho[0:1])[0]

    

    def calc_f_out(self):
        for i in range(9):
            feq = self.calc_f_eq_i(i, self.u, self.rho)
            self.f_out[i] = self.f_in[i] - self.w * (self.f_in[i] - feq)

    def bounce_back(self):
        mask = np.zeros((self.nx, self.ny), dtype=bool)
        mask[self.obstacle['x'], self.obstacle['y']] = True
        for i in range(9):
            self.f_out[8 - i, mask] = self.f_in[i, mask]
        self.u[mask] = 0.0

    def collision_and_stream(self):
        for i in range(9):
            self.f_in[i] = np.roll(
                np.roll(self.f_out[i], int(self.v[i, 0]), axis=0),
                int(self.v[i, 1]), axis=1
            )

    

    def _save_snapshot(self, step):
        u_mag = np.sqrt(self.u[...,0]**2 + self.u[...,1]**2)
        self.field_u.append(u_mag.copy())
        self.field_den.append(self.rho.copy())
        self.field_p.append(self.rho / 3)
        self.field_steps.append(step)

    

    def solve(self, n_step=1000, step_frame=20):
        for step in range(n_step):
            self.calc_outflow(self.f_in)
            self.rho = np.sum(self.f_in, axis=0)
            self.u = self.calc_u(self.rho, self.f_in, self.v)

            u_mag = np.sqrt(self.u[...,0]**2 + self.u[...,1]**2)
            mask = u_mag > 0.15
            self.u[mask] *= (0.15 / u_mag[mask])[:, None]

            self.calc_inflow()
            self.calc_f_out()
            self.bounce_back()
            self.collision_and_stream()

            if step % step_frame == 0:
                self._save_snapshot(step)

        print("✅ Симуляция завершена")

    

    def create_animation(self, field_type='velocity', save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        data = np.array(self.field_u)
        steps = self.field_steps

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(data[0].T, cmap='plasma', origin='lower', aspect='auto')
        txt = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes,
                      fontsize=12, bbox=dict(facecolor='black', alpha=0.6))

        def update(frame):
            im.set_array(data[frame].T)
            txt.set_text(f"Iteration: {steps[frame]}")
            return [im, txt]

        ani = animation.FuncAnimation(fig, update, frames=len(data), blit=True)

        if save_path is None:
            save_path = f'variant_{self.N}_velocity.gif'

        ani.save(save_path, writer='pillow', fps=10, dpi=100)
        plt.close(fig)
        print(f"✅ GIF сохранён: {save_path}")



    def save_snapshots_png(self, steps=(100,200,300,400,500,600,700,800,900)):
        import matplotlib.pyplot as plt

        os.makedirs("snapshots", exist_ok=True)

        for i, step in enumerate(self.field_steps):
            if step in steps:
                plt.figure(figsize=(6, 3))
                plt.imshow(self.field_u[i].T, cmap='plasma', origin='lower')
                plt.title(f"Velocity magnitude, step {step}")
                plt.colorbar()
                plt.savefig(f"snapshots/velocity_{step}.png", dpi=200)
                plt.close()

        print("✅ PNG-снимки сохранены")



if __name__ == "__main__":
    params = {'nx': 420, 'ny': 180, 'N': 10}
    obstacle = {'xc': 105, 'yc': 90, 'r': 20}

    model = LatticeGas(params, obstacle)
    model.solve(n_step=1000, step_frame=20)
    model.create_animation()
    model.save_snapshots_png()

