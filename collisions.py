from manimlib.imports import *

class Box(Scene):
    def construct(self):
        box = Rectangle()
        box.scale(3)
        self.box = box
        self.particles = []
        self.dots = []
        positions = [[i+0.5, j+0.75, 0.0] for j in range(-3,4) for i in range(-6, 6)]

        for i in range(40):
            p = Particle(pos_in=np.array(positions[i]))
            dot = Dot(radius=p.rad, color=p.color)
            self.particles.append(p)
            self.dots.append(dot)
            dot.move_to(p.pos)

        self.play(ShowCreation(box), *[ShowCreation(d) for d in self.dots], run_time=1)
        self.dt = 1 / 60
        count_frames = 600
        for frame in range(count_frames):
            self.update()
            for d, p in zip(self.dots, self.particles):
                d.move_to(p.pos)
            self.wait(self.dt)
            self.printProgressBar(frame, count_frames - 1)

    def update(self):
        for p in self.particles:
            p.pf0 = np.copy(p.pos)
            for i in range(2):
                p.pos[i] += p.vel[i] * self.dt
                p.vel[i] += p.accel[i] * self.dt
            p.pf1 = np.copy(p.pos)
        self.check_possible_collisions()

    def check_possible_collisions(self):
        self.check_wall_collisions()
        checked = []
        for i in self.particles:
            for j in self.particles:
                if i == j or set((i, j)) in checked:
                    continue
                checked.append(set((i, j)))
                self.handle_collisions(i, j)

    def handle_collisions(self, p1, p2):
        def compute_vel(x1, x2, v1, v2, m1, m2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)
        v1 = p1.vel
        v2 = p2.vel
        m1 = p1.mass
        m2 = p2.mass
        x1 = p1.pos
        x2 = p2.pos
        rad1 = p1.rad
        rad2 = p2.rad

        if np.linalg.norm(x1 - x2) > rad1 + rad2:
            return
        tc = self.get_tc(p1, p2)
        if tc is None:
            return

        tmp_v1 = compute_vel(x1, x2, v1, v2, m1, m2)
        tmp_v2 = compute_vel(x2, x1, v2, v1, m2, m1)

        inter_p1 = self.get_interpolation(p1, tc)
        inter_p2 = self.get_interpolation(p2, tc)

        p1.vel = tmp_v1
        p2.vel = tmp_v2

        for i in range(2):
            p1.pos[i] = inter_p1[i] + p1.vel[i] * (self.dt * (1 - tc))
            p2.pos[i] = inter_p2[i] + p2.vel[i] * (self.dt * (1 - tc))

    def get_tc(self, p1, p2):
        P = p2.pf0 - p1.pf0
        V = p2.vel - p1.vel
        r = p1.rad + p2.rad
        coeff = [np.dot(V, V), 2*np.dot(P, V), np.dot(P,P) - r**2]
        roots = [n for n in np.roots(coeff) if not isinstance(n, complex)]
        if roots:
            return min(roots)
        return None

    def check_wall_collisions(self):
        for p in self.particles:
            box_left = self.box.get_left()[0]
            box_right = self.box.get_right()[0]
            box_top = self.box.get_top()[1]
            box_bottom = self.box.get_bottom()[1]

            tc = None

            if p.left()[0] <= box_left:
                tc = (box_left + p.rad - p.pf0[0]) / (p.pf1[0] - p.pf0[0])
                p.vel[0] = -p.vel[0]
            elif p.right()[0] >= box_right:
                tc = (box_right - p.rad - p.pf0[0]) / (p.pf1[0] - p.pf0[0])
                p.vel[0] = -p.vel[0]
            if p.top()[1] >= box_top:
                tc = (box_top - p.rad - p.pf0[1]) / (p.pf1[1] - p.pf0[1])
                p.vel[1] = -p.vel[1]
            elif p.bottom()[1] <= box_bottom:
                tc = (box_bottom + p.rad - p.pf0[1]) / (p.pf1[1] - p.pf0[1])
                p.vel[1] = -p.vel[1]              

            if tc is not None:
                interpolation = self.get_interpolation(p, tc)
                for i in range(2):
                    p.pos[i] = interpolation[i] + p.vel[i] * (self.dt * (1-tc))

    def get_interpolation(self, p, t):
        return np.array([t * p.pf1[0] + (1 - t) * p.pf0[0], t * p.pf1[1] + (1 - t) * p.pf0[1], 0.0])

    # Progress Bar function by Greenstick https://stackoverflow.com/a/34325723/14099362
    def printProgressBar (self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        if iteration == total: 
            print()

class Particle():
    def __init__(self, pos_in=None, vel_in=None):
        random.seed()
        self.rad = random.uniform(0.05, 0.2)    
        self.pos = np.array([random.uniform(-5.0, 5.0), random.uniform(-2.5, 2.5), 0.0]) if pos_in is None else pos_in                         
        self.vel = np.random.uniform(low=-2.5, high=2.5, size=(3,)) if vel_in is None else vel_in                               
        self.accel = np.array([0.0, 0.0, 0.0])                                  
        self.mass = self.rad*math.pi**2
        self.pf0 = np.copy(self.pos)
        self.pf1 = None                           
        self.color = BLUE if random.randint(1, 2) == 1 else RED

    def right(self):                               
        return [self.pos[0] + self.rad, self.pos[1]]

    def left(self):
        return [self.pos[0] - self.rad, self.pos[1]]

    def top(self):
        return [self.pos[0], self.pos[1] + self.rad]
    
    def bottom(self):
        return [self.pos[0], self.pos[1] - self.rad]