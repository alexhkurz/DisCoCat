# https://docs.discopy.org/en/main/notebooks/qnlp.html

from discopy.symmetric import Ty, Box, Id, Swap, Diagram, Functor
from discopy.drawing import Equation

## Drawing cooking recipes

print(Ty('sentence', 'qubit'))

egg, white, yolk = Ty(*['egg', 'white', 'yolk'])

assert egg @ (white @ yolk) == (egg @ white) @ yolk  # associativity
assert egg @ Ty() == egg == Ty() @ egg               # unitality

crack = Box(name='crack', dom=egg, cod=white @ yolk)

crack.draw(figsize=(2, 2))

mix = Box('mix', white @ yolk, egg)

crack_tensor_mix = crack @ mix
crack_then_mix = crack >> mix

Equation(crack_tensor_mix, crack_then_mix, symbol=' and ').draw(space=2, figsize=(8, 2))

assert crack >> Id(white @ yolk) == crack == Id(egg) >> crack
assert crack @ Id() == crack == Id() @ crack

assert crack @ egg == crack @ Id(egg)
assert egg @ crack == Id(egg) @ crack

sugar, yolky_paste = Ty('sugar'), Ty('yolky_paste')
beat = Box('beat', yolk @ sugar, yolky_paste)

crack_then_beat = crack @ sugar >> white @ beat

crack_then_beat.draw(figsize=(3, 2))

merge = lambda x: Box('merge', x @ x, x)

crack_two_eggs = crack @ crack\
    >> white @ Swap(yolk, white) @ yolk\
    >> merge(white) @ merge(yolk)

crack_two_eggs.draw(figsize=(3, 4))

assert crack_two_eggs == Diagram.decode(
    dom=egg @ egg, boxes_and_offsets=[
        (crack,             0),
        (crack,             2),
        (Swap(yolk, white), 1),
        (merge(white),      0),
        (merge(yolk),       1)])

crack2 = Box("crack", egg @ egg, white @ yolk)

open_crack2 = Functor(
    ob=lambda x: x,
    ar={crack2: crack_two_eggs, beat: beat})

crack2_then_beat = crack2 @ Id(sugar) >> Id(white) @ beat

Equation(crack2_then_beat, open_crack2(crack2_then_beat),
         symbol='$\\mapsto$').draw(figsize=(7, 3.5))

oeuf, blanc, jaune, sucre = Ty("oeuf"), Ty("blanc"), Ty("jaune"), Ty("sucre")

ouvrir = Box("ouvrir", oeuf, blanc @ jaune)
battre = Box("battre", jaune @ sucre, jaune)

english2french = Functor(
    ob={egg: oeuf,
        white: blanc,
        yolk: jaune,
        sugar: sucre,
        yolky_paste: jaune},
    ar={crack: ouvrir,
        beat: battre})

english2french(crack_then_beat).draw(figsize=(3, 2))

echanger = lambda x, y: Box("échanger", x @ y, y @ x, draw_as_wires=True)
melanger = lambda x: Box("mélanger", x @ x, x)

for x in [white, yolk]:
    english2french.ar[merge(x)] = melanger(english2french(x))

english2french(open_crack2(crack2_then_beat)).draw(figsize=(4, 4))

## Tensor as boxes

from discopy.tensor import Dim, Tensor

matrix = Tensor([0, 1, 1, 0], Dim(2), Dim(2))

matrix.array

