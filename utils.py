import numpy as np
import os
import random
import re

qwertyKeyboardArray = [
    ['`','1','2','3','4','5','6','7','8','9','0','-','='],
    ['q','w','e','r','t','y','u','i','o','p','ă','î','â'],
    ['a','s','d','f','g','h','j','k','l','ș','ț'],
    ['z','x','c','v','b','n','m',',','.','/'],
    ['', '', ' ', ' ', ' ', ' ', ' ', '', '']
    ]

qwertyShiftedKeyboardArray = [
    ['~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'Ă', 'Î', 'Â'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Ș', 'Ț'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    ['', '', ' ', ' ', ' ', ' ', ' ', '', '']
    ]

def multiple_is(s):
    if len(s) < 2:
        return s

    if s[-2:] == 'ii':
        chance = random.randint(0, 1)
        if chance:
            return s[:-1]

        return s + 'i'

    if s[-1] == 'i':
        chance = random.randint(0, 1)

        if chance:
            return s

        return s + 'i'

    return s


def replace(s):
    if len(s) < 2:
        return s
    r = random.randint(0, len(s) - 1)
    if not s[r].isalpha():
        return s

    candidates = get_neighbours(s[r])
    return s[:r] + random.choice(candidates) + s[r + 1:]


def insert(s):
    if len(s) < 2:
        return s
    r = random.randint(0, len(s) - 1)
    if not s[r].isalpha():
        return s

    candidates = get_neighbours(s[r])
    chance = random.randint(0, 1)
    return s[:r + chance] + random.choice(candidates) + s[r + chance:]

def doublle(s):
    if len(s) < 2:
        return s

    r = random.randint(0, len(s) - 1)

    if not s[r].isalpha():
        return s

    return s[:r] + s[r] + s[r:]

def delete(s):
    if len(s) < 2:
        return s
    r = random.randint(0, len(s) - 1)

    if not s[r].isalpha():
        return s

    return s[:r] + s[r + 1:]

def get_keyboard(c):
    if (True in [c in r for r in qwertyKeyboardArray]):
        return qwertyKeyboardArray
    elif (True in [c in r for r in qwertyShiftedKeyboardArray]):
        return qwertyShiftedKeyboardArray

    raise ValueError(c + "not found in any keyboard layouts")

def get_coord(c, keyboard):
    row = -1
    column = -1
    for r in keyboard:
        if c in r:
            row = keyboard.index(r)
            column = r.index(c)
            return row, column
    raise ValueError(c + "not found in given keyboard layout")

def get_neighbours(c):
    keyboard = get_keyboard(c)
    x, y = get_coord(c, keyboard)

    if y < 1:
        neighbours = keyboard[x-1:x+2]
        neighbours = [neighbours[i][y:y+2] for i in range(len(neighbours))]

    elif x < 1:
        neighbours = keyboard[x:x+2]
        neighbours = [neighbours[i][y-1:y+2] for i in range(len(neighbours))]

    else:
        neighbours = keyboard[x-1:x+2]
        neighbours = [neighbours[i][y-1:y+2] for i in range(len(neighbours))]

    neighbours = [n for sub_neighbor in neighbours for n in sub_neighbor]

    neighbours = [c for c in neighbours if c.isalpha()]
    return neighbours


def add_noise(text, prob=0.4):
    operations = [
        multiple_is,
        doublle,
        replace,
        insert,
        delete
    ]

    words = []

    for word in text.split(' '):
        chance = random.random()

        if chance < prob:
            operation = random.choice(operations)
            word = operation(word)

        words.append(word)

    return ' '.join(words)

