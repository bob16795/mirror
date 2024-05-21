import cv2
import pygame
from pygame.locals import *
import numpy as np
import random
import os

TOTAL_FACE_TIME = 0.5
FACE_FACE_TIME = 0.1

#vid = cv2.VideoCapture("lol.mp4")
vid = cv2.VideoCapture(2)
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
# screen = pygame.display.set_mode([1280,720], pygame.FULLSCREEN)
screen = pygame.display.set_mode([1280,720])

running = True

face_imgs = []

for img in os.listdir("faces"):
    face_imgs += [pygame.image.load(os.path.join('faces', img))]

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []

clock = pygame.time.Clock()

while running:
    ret, frame = vid.read()

    dt = clock.tick(60)

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.01, minNeighbors=6, minSize=(100, 100)
    )

    for f1 in face:
        match = -1
        center1 = (f1[0] + f1[2] * 0.5, f1[1] + f1[3] * 0.5)
        for idx, f2 in enumerate(faces):
            center2 = (f2["face"][0] + f2["face"][2] * 0.5, f2["face"][1] + f2["face"][3] * 0.5)
            bad = False
            for i in range(2):
                if abs(center1[i] - center2[i]) > f2["face"][2] * 2.0:
                    bad = True
            if not bad:
                match = idx 
                break
        if match == -1:
            faces += [{"face": f1, "targ": f1, "time": TOTAL_FACE_TIME, "img": random.sample(face_imgs, 1)[0]}]
        else:
            faces[match] = {"face": faces[match]["face"], "targ": f1, "time": TOTAL_FACE_TIME, "img": faces[match]["img"]}

    for f in faces:
        f["time"] -= dt / 1000.0
        for i in range(4):
            f["face"][i] += (f["targ"][i] - f["face"][i]) * np.clip(dt / 1000.0 * 6.0, 0, 1)

    faces = [face for face in faces if face["time"] > 0]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)

    ratio = frame.get_width() / frame.get_height()
    (width, height) = pygame.display.get_window_size()
    scale = height / frame.get_height()

    frame = pygame.transform.scale(frame, (scale * frame.get_width(), scale * frame.get_height()))
    
    offsetx = (width - frame.get_width()) * 0.5 

    screen.fill((255, 255, 255, 255))
    
    screen.blit(frame, (offsetx,0))

    for f in faces:
        (x, y, w, h) = f["face"]

        img = pygame.transform.scale(f["img"], (w * scale, h * scale))
        img.set_alpha(np.clip(f["time"] / FACE_FACE_TIME * 255, 0, 255))

        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(frame.get_width() - x * scale - h * scale + offsetx, y * scale, w * scale, h * scale), 5)

        screen.blit(img, (frame.get_width() - x * scale - h * scale + offsetx, y * scale))

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
