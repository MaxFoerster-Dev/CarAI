import os
import pygame
import random
import math
import numpy as np

# Pygame initialisieren
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Racer - Evolution (8-32-32-6)")
clock = pygame.time.Clock()

# Farben
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)

# Hyperparameter
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
MUTATION_SCALE = 0.2
BRAIN_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_brain.npz")

# --- NEURONALES NETZWERK (8-32-32-6) ---
class NeuralNetwork:
    def __init__(self, weights=None):
        if weights is None:
            # 8 Inputs -> 32 Hidden1 -> 32 Hidden2 -> 6 Outputs
            self.w1 = np.random.randn(8, 32)
            self.b1 = np.zeros((1, 32))
            self.w2 = np.random.randn(32, 32)
            self.b2 = np.zeros((1, 32))
            self.w3 = np.random.randn(32, 6)
            self.b3 = np.zeros((1, 6))
        else:
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = weights

    def relu(self, x):
        return np.maximum(0, x)

    def copy(self):
        return NeuralNetwork([
            self.w1.copy(), self.b1.copy(),
            self.w2.copy(), self.b2.copy(),
            self.w3.copy(), self.b3.copy()
        ])

    def save(self, path=BRAIN_SAVE_PATH):
        np.savez(path,
                 w1=self.w1,
                 b1=self.b1,
                 w2=self.w2,
                 b2=self.b2,
                 w3=self.w3,
                 b3=self.b3)

    @staticmethod
    def load(path=BRAIN_SAVE_PATH):
        if not os.path.exists(path):
            return None

        data = np.load(path)
        return NeuralNetwork([
            data["w1"], data["b1"],
            data["w2"], data["b2"],
            data["w3"], data["b3"]
        ])

    def predict(self, inputs):
        # Forward Pass
        x = np.array(inputs).reshape(1, -1)
        h1 = self.relu(np.dot(x, self.w1) + self.b1)
        h2 = self.relu(np.dot(h1, self.w2) + self.b2)
        out = np.dot(h2, self.w3) + self.b3  # Linear Output
        return np.argmax(out)

    def mutate(self):
        def mutate_array(arr):
            mask = np.random.rand(*arr.shape) < MUTATION_RATE
            arr[mask] += np.random.randn(*arr[mask].shape) * MUTATION_SCALE
            return arr

        return NeuralNetwork([
            mutate_array(self.w1.copy()), mutate_array(self.b1.copy()),
            mutate_array(self.w2.copy()), mutate_array(self.b2.copy()),
            mutate_array(self.w3.copy()), mutate_array(self.b3.copy())
        ])

# --- AUTO KLASSE ---
class Car:
    def __init__(self, brain=None, track=None):
        self.x = WIDTH // 2
        self.y = HEIGHT - 100
        self.angle = -90  # Schaut nach oben
        self.speed = 4    # etwas Grundspeed, damit Start nicht "tot" ist
        self.max_speed = 6
        self.alive = True
        self.distance_traveled = 0
        self.fitness = 0
        self.brain = brain if brain else NeuralNetwork()
        self.sensors = [0.0] * 7

    def update(self, track):
        if not self.alive:
            return

        # 1. Sensoren updaten (Raycasting)
        self.cast_tech_sensors(track)
        
        # 2. KI entscheidet
        # Inputs: 7 Sensoren + 1 Geschwindigkeit (normalisiert)
        inputs = self.sensors + [self.speed / self.max_speed]
        action = self.brain.predict(inputs)

        # 6 Aktionen: 0=Gas, 1=Bremse, 2=Links, 3=Rechts, 4=Geradeaus, 5=Nichts
        if action == 0:
            self.speed = min(self.speed + 0.1, self.max_speed)
        elif action == 1:
            self.speed = max(self.speed - 0.1, 1)
        elif action == 2:
            self.angle -= 4
        elif action == 3:
            self.angle += 4

        # Bewegung berechnen
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.speed
        dy = math.sin(rad) * self.speed
        
        self.x += dx
        # Da die Strecke nach unten scrollt, addieren wir den Fortschritt relativ zur Welt
        self.distance_traveled += abs(dy)

        # Belohnung nur für echten Vorwärtsdrang, nicht nur für Seitwärtsbewegung
        if dy < 0:
            self.fitness += abs(dy) * 2.0
        else:
            self.fitness -= 1.0

        # Bonus für die Spurmitte: ähnliche Werte links/rechts = stabiler Kurs
        sensor_diff = abs(self.sensors[0] - self.sensors[6])
        self.fitness += (1.0 - sensor_diff) * 0.5

        # Kollisionsprüfung mit den Wänden
        if track.check_collision(self.x, self.y):
            self.alive = False

    def cast_tech_sensors(self, track):
        # 7 Winkel relativ zur Ausrichtung des Autos
        sensor_angles = [-90, -45, -20, 0, 20, 45, 90]
        max_ray_len = 200

        for i, angle_offset in enumerate(sensor_angles):
            rad = math.radians(self.angle + angle_offset)
            dist = max_ray_len
            
            # Strahl schrittweise vortasten
            for step in range(5, max_ray_len, 5):
                target_x = self.x + math.cos(rad) * step
                target_y = self.y + math.sin(rad) * step
                if track.check_collision(target_x, target_y):
                    dist = step
                    break
            
            # Normalisiert zwischen 0.1 und 1.0 (Wand in der Nähe -> hoher Wert,
            # freie Bahn -> kleiner Wert, aber nicht komplett 0, damit das Netz nicht "tot" wird)
            self.sensors[i] = 0.1 + 0.9 * (1.0 - (dist / max_ray_len))

    def draw(self, surface):
        if not self.alive:
            return
        # Auto zeichnen
        pygame.draw.circle(surface, GREEN, (int(self.x), int(self.y)), 5)
        
        # Blickrichtung (kleiner Zeiger)
        rad = math.radians(self.angle)
        pygame.draw.line(surface, RED, (self.x, self.y), (self.x + math.cos(rad)*15, self.y + math.sin(rad)*15), 2)


def draw_input_axes(surface, car):
    if not car.alive:
        return

    cx, cy = int(car.x), int(car.y)
    values = car.sensors + [car.speed / car.max_speed]
    axis_count = len(values)

    for i, value in enumerate(values):
        angle = math.radians(-90 + i * (360 / axis_count))
        length = 20 + value * 90
        end_x = cx + math.cos(angle) * length
        end_y = cy + math.sin(angle) * length
        pygame.draw.line(surface, BLUE, (cx, cy), (end_x, end_y), 1)
        pygame.draw.circle(surface, BLUE, (int(end_x), int(end_y)), 2)

# --- PROZEDURALE SCROLLENDE STRECKE ---
class DynamicTrack:
    def __init__(self):
        self.scroll_y = 0
        self.points = []
        self.current_center = WIDTH // 2
        self.current_width = 160
        self.target_center = self.current_center
        self.phase = 0.0

        for y in range(HEIGHT + 200, -200, -5):
            center = WIDTH // 2 + math.sin(y / 150.0) * 180
            width = 150 + math.sin(y / 90.0) * 25
            self.points.append([y, center, width])

        self.distance_metric = 0

    def update(self, scroll_speed):
        self.scroll_y += scroll_speed
        self.distance_metric += scroll_speed
        self.phase += 0.02 + scroll_speed * 0.01

        if self.distance_metric >= 220:
            self.distance_metric = 0
            self.target_center = WIDTH // 2 + random.randint(-180, 180) + math.sin(self.phase) * 80
            self.current_width = random.randint(120, 190)

        self.current_center += (self.target_center - self.current_center) * 0.04

        for p in self.points:
            p[0] += scroll_speed
            p[1] += (self.current_center - p[1]) * 0.08
            p[2] += (self.current_width - p[2]) * 0.08

        self.points = [p for p in self.points if p[0] < HEIGHT + 120]

        while self.points[-1][0] > -100:
            new_y = self.points[-1][0] - 5
            new_center = WIDTH // 2 + math.sin((new_y + self.scroll_y) / 160.0) * 180
            new_width = 120 + 35 * math.sin((new_y + self.scroll_y) / 100.0)
            self.points.append([new_y, new_center, new_width])

    def check_collision(self, x, y):
        closest_p = min(self.points, key=lambda p: abs(p[0] - y))
        _, center_x, width = closest_p

        if x < (center_x - width // 2) or x > (center_x + width // 2):
            return True
        return False

    def draw(self, surface):
        for i in range(len(self.points) - 1):
            y1, c1, w1 = self.points[i]
            y2, c2, w2 = self.points[i + 1]

            pygame.draw.line(surface, WHITE, (c1 - w1 // 2, y1), (c2 - w2 // 2, y2), 3)
            pygame.draw.line(surface, WHITE, (c1 + w1 // 2, y1), (c2 + w2 // 2, y2), 3)

            pygame.draw.line(surface, GRAY, (c1, y1), (c2, y2), 1)

# --- EVOLUTIONÄRER MANAGER ---
class EvolutionManager:
    def __init__(self, track=None):
        self.generation = 1

        saved_brain = NeuralNetwork.load(BRAIN_SAVE_PATH)
        if saved_brain is not None:
            self.cars = [Car(brain=saved_brain.copy(), track=track)] + [Car(track=track) for _ in range(POPULATION_SIZE - 1)]
        else:
            self.cars = [Car(track=track) for _ in range(POPULATION_SIZE)]

    def all_dead(self):
        return all(not car.alive for car in self.cars)

    def evolve(self, track=None):
        # Sortiere nach Fitness (beste KIs zuerst)
        self.cars.sort(key=lambda c: c.fitness, reverse=True)
        best_brain = self.cars[0].brain
        best_brain.save(BRAIN_SAVE_PATH)

        new_cars = []
        # Der Beste überlebt unverändert (Elitismus)
        new_cars.append(Car(brain=best_brain.copy(), track=track))

        # Rest auffüllen mit Mutationen des Besten
        for _ in range(POPULATION_SIZE - 1):
            new_cars.append(Car(brain=best_brain.mutate(), track=track))

        self.cars = new_cars
        self.generation += 1

# --- MAIN LOOP ---
def main():
    track = DynamicTrack()
    evo_manager = EvolutionManager(track)
    font = pygame.font.SysFont("Arial", 20)

    running = True
    while running:
        clock.tick(60)
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        time_scale = 5.0 if keys[pygame.K_f] else 1.0

        # Finde die Geschwindigkeit des am weitesten vorne liegenden, lebenden Autos
        alive_cars = [c for c in evo_manager.cars if c.alive]
        scroll_speed = max([c.speed for c in alive_cars]) if alive_cars else 0

        # Strecke updaten und scrollen
        track.update(scroll_speed * time_scale)
        track.draw(screen)

        # Autos updaten und zeichnen
        for car in evo_manager.cars:
            car.update(track)
            car.draw(screen)
            draw_input_axes(screen, car)

        # Evolution triggern, wenn alle gecrasht sind
        if evo_manager.all_dead():
            track = DynamicTrack() # Strecke für die neue Generation zurücksetzen
            evo_manager.evolve(track)

        # UI Text anzeigen
        text_gen = font.render(f"Generation: {evo_manager.generation}", True, BLUE)
        text_alive = font.render(f"Lebend: {len(alive_cars)} / {POPULATION_SIZE}", True, BLUE)
        text_dist = font.render(f"Max Distanz: {int(max([c.distance_traveled for c in evo_manager.cars]))}", True, BLUE)
        
        screen.blit(text_gen, (10, 10))
        screen.blit(text_alive, (10, 35))
        screen.blit(text_dist, (10, 60))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
