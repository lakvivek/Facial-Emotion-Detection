import pygame as pg
import cv2
import os, random
import sys,os


file_list = []
for root, dirs, files in os.walk("."):
    for filename in files:
        file_list.append(filename)


def play_music(detection, volume=0.8):

    if detection == 5:
        print("I'm happy")
        x = []
        for k in file_list:
            if k.startswith('happy'):
                x.append(k)

        print (random.choice(x))
        music_file = random.choice(x)

    else:
        print("I'm not happy")
        x = []
        for k in file_list:
            if k.startswith('neutral'):
                x.append(k)

        print(random.choice(x))
        music_file = random.choice(x)

    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 2048
    pg.mixer.init(freq, bitsize, channels, buffer)
    pg.mixer.music.set_volume(volume)

    try:
        pg.mixer.music.load(music_file)
        print("Music file {} loaded!".format(music_file))
    except pg.error:
        print("File {} not found! ({})".format(music_file, pg.get_error()))
        return
    clock = pg.time.Clock()
    # clock
    start_ticks = pg.time.get_ticks()
    pg.mixer.music.play()
    # while pg.mixer.music.get_busy():
    #     clock.tick(5)
    #     # return True
    # starter tick
    seconds = 0
    while pg.mixer.music.get_busy():
        seconds = (pg.time.get_ticks() - start_ticks) / 1000  # calculate how many seconds
        clock.tick(30)
        if seconds > 30:
            break# if more than 10 seconds close the game

        print(seconds)



def stop():
    # clock.tick(30)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     pg.mixer.music.stop()
    pg.mixer.music.stop()


if __name__ == '__main__':
    #music_file = "happy1.mp3"

    detection = 5
    y= play_music(detection)
    # print(y)
