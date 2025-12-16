

import numpy as np


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

        self.w = 1.0 / (3.0 * self.nu + 0.5)
        self.w = min(self.w, 1.9)


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

        self._initialize_fields()



    @staticmethod
    def _calculate_params_from_N(N):
        u_lb_min, u_lb_max = 0.01, 0.1
        Re_min, Re_max = 20, 1000

        u_lb = u_lb_min + 2 * (u_lb_max - u_lb_min) / (N - 1)
        Re = int(Re_min + 2 * (Re_max - Re_min) / (N - 1))
        return u_lb, Re



    def _initialize_fields(self):
        self.u[:, :, 0] = self.u_lb
        self.u[:, :, 1] = 0.0
        self.rho[:, :] = 1.0

        for i in range(9):
            self.f_in[i] = self.calc_f_eq_i(i, self.u, self.rho)



    @staticmethod
    def add_cylinder(xc, yc, r, shape):
        nx, ny = shape
        x, y = np.ogrid[:nx, :ny]
        mask = (x - xc)**2 + (y - yc)**2 <= r**2

        if xc - r <= 0 or xc + r >= nx - 1 or yc - r <= 0 or yc + r >= ny - 1:
            raise ValueError("Цилиндр выходит за границы домена")

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

        with np.errstate(divide='ignore', invalid='ignore'):
            ux /= rho
            uy /= rho

        return np.stack([np.nan_to_num(ux), np.nan_to_num(uy)], axis=-1)



    def calc_f_eq_i(self, i, u, rho):
        v_dot_u = self.v[i, 0] * u[:, :, 0] + self.v[i, 1] * u[:, :, 1]
        u_sqr = u[:, :, 0]**2 + u[:, :, 1]**2

        return rho * self.alpha[i] * (
            1 + 3*v_dot_u + 4.5*v_dot_u**2 - 1.5*u_sqr
        )



    def calc_inflow(self):
        self.u[0, :, 0] = self.u_lb
        self.u[0, :, 1] = 0.0

        rho_known = np.sum(self.f_in[[0,1,2,3,4,5], 0, :], axis=0)
        self.rho[0, :] = rho_known / (1 - self.u[0, :, 0])
        self.rho[0, :] = np.maximum(self.rho[0, :], 1e-6)

        for i in [6,7,8]:
            self.f_in[i, 0, :] = self.calc_f_eq_i(
                i, self.u[0:1], self.rho[0:1]
            )[0]



    def calc_f_out(self):
        for i in range(9):
            f_eq = self.calc_f_eq_i(i, self.u, self.rho)
            self.f_out[i] = self.f_in[i] - self.w * (self.f_in[i] - f_eq)

    def bounce_back(self):
        mask = np.zeros((self.nx, self.ny), dtype=bool)
        mask[self.obstacle['x'], self.obstacle['y']] = True

        for i in range(9):
            self.f_out[8 - i, mask] = self.f_in[i, mask]

        self.u[mask, :] = 0.0

    def collision_and_stream(self):
        for i in range(9):
            self.f_in[i] = np.roll(
                np.roll(self.f_out[i], int(self.v[i, 0]), axis=0),
                int(self.v[i, 1]), axis=1
            )



    def _save_snapshot(self):
        u_mag = np.sqrt(self.u[:, :, 0]**2 + self.u[:, :, 1]**2)
        self.field_u.append(u_mag.copy())
        self.field_den.append(self.rho.copy())
        self.field_p.append(self.rho / 3.0)



    def solve(self, n_step=1000, step_frame=20):
        print(f"🚀 Старт симуляции (ω={self.w:.3f}, ν={self.nu:.5f})")

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
                self._save_snapshot()

        print("✅ Симуляция завершена")



    def create_animation(self, field_type='velocity', save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        if field_type == 'density':
            data = np.array(self.field_den)
            title, cmap = 'Плотность', 'viridis'
        elif field_type == 'velocity':
            data = np.array(self.field_u)
            title, cmap = 'Скорость', 'plasma'
        elif field_type == 'pressure':
            data = np.array(self.field_p)
            title, cmap = 'Давление', 'coolwarm'
        else:
            print("Неизвестный тип поля")
            return

        if len(data) == 0:
            print("Нет данных для анимации")
            return

        if save_path is None:
            save_path = f'variant_{self.N}_{field_type}.gif' if self.N else f'{field_type}.gif'

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(data[0].T, cmap=cmap, origin='lower', aspect='auto')

        plt.colorbar(im, ax=ax)
        ax.set_title(title)

        def update(frame):
            im.set_array(data[frame].T)
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(data), blit=True)
        ani.save(save_path, writer='pillow', fps=10, dpi=100)

        plt.close(fig)
        print(f"✅ GIF сохранён: {save_path}")



if __name__ == "__main__":
    params = {'nx': 420, 'ny': 180, 'N': 10}
    obstacle = {'xc': 105, 'yc': 90, 'r': 20}

    model = LatticeGas(params, obstacle)
    model.solve(n_step=1000, step_frame=20)
    model.create_animation('velocity')
