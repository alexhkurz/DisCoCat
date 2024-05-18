# https://docs.discopy.org/en/main/notebooks/qnlp.html

from discopy.symmetric import Ty, Box, Id, Swap, Diagram, Functor
from discopy.drawing import Equation
from discopy.tensor import Dim, Tensor
import numpy as np

## Drawing cooking recipes
## Drawing cooking recipes
## Drawing cooking recipes
## Drawing cooking recipes
## Drawing cooking recipes

print(Ty('sentence', 'qubit'))

egg, white, yolk = Ty(*['egg', 'white', 'yolk'])

assert egg @ (white @ yolk) == (egg @ white) @ yolk  # associativity
assert egg @ Ty() == egg == Ty() @ egg               # unitality

crack = Box(name='crack', dom=egg, cod=white @ yolk)

# crack.draw(figsize=(2, 2))

mix = Box('mix', white @ yolk, egg)

crack_tensor_mix = crack @ mix
crack_then_mix = crack >> mix

# Equation(crack_tensor_mix, crack_then_mix, symbol=' and ').draw(space=2, figsize=(8, 2))

assert crack >> Id(white @ yolk) == crack == Id(egg) >> crack
assert crack @ Id() == crack == Id() @ crack

assert crack @ egg == crack @ Id(egg)
assert egg @ crack == Id(egg) @ crack

sugar, yolky_paste = Ty('sugar'), Ty('yolky_paste')
beat = Box('beat', yolk @ sugar, yolky_paste)

crack_then_beat = crack @ sugar >> white @ beat

# crack_then_beat.draw(figsize=(3, 2))

merge = lambda x: Box('merge', x @ x, x)

crack_two_eggs = crack @ crack\
    >> white @ Swap(yolk, white) @ yolk\
    >> merge(white) @ merge(yolk)

# crack_two_eggs.draw(figsize=(3, 4))

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

# Equation(crack2_then_beat, open_crack2(crack2_then_beat),
#          symbol='$\\mapsto$').draw(figsize=(7, 3.5))

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

# english2french(crack_then_beat).draw(figsize=(3, 2))

echanger = lambda x, y: Box("échanger", x @ y, y @ x, draw_as_wires=True)
melanger = lambda x: Box("mélanger", x @ x, x)

for x in [white, yolk]:
    english2french.ar[merge(x)] = melanger(english2french(x))

# english2french(open_crack2(crack2_then_beat)).draw(figsize=(4, 4))

## Tensor as boxes
## Tensor as boxes
## Tensor as boxes
## Tensor as boxes
## Tensor as boxes

from discopy.tensor import Cup, Cap, Id # tensor.Id is different from symmetric.Id (!!)
from discopy import tensor

matrix = Tensor([0, 1, 1, 0], Dim(2), Dim(2))
matrix.array

assert matrix >> Tensor.id(Dim(2)) == matrix == Tensor.id(Dim(2)) >> matrix
vector = Tensor([0, 1], Dim(1), Dim(2))
vector >> matrix

assert Tensor.id(Dim(1)) @ matrix == matrix == matrix @ Tensor.id(Dim(1))
Tensor.id(Dim(1))

print(vector @ vector)
print(vector @ matrix)

assert np.all(
    (matrix >> matrix).array == matrix.array.dot(matrix.array))
assert np.all(
    (matrix @ matrix).array == np.moveaxis(np.tensordot(
    matrix.array, matrix.array, 0), range(4), [0, 2, 1, 3]))

matrix = Tensor[complex]([0, -1j, 1j, 0], Dim(2), Dim(2))
matrix >> matrix.dagger()

vector1 = Tensor[complex]([-1j, 1j], Dim(1), Dim(2))
vector.cast(complex) >> vector1.dagger()

print(vector + vector)

zero = Tensor.zero(Dim(1), Dim(2))
assert vector + zero == vector == zero + vector

swap = Tensor.swap(Dim(2), Dim(3))
assert swap.dom == Dim(2) @ Dim(3) and swap.cod == Dim(3) @ Dim(2)
assert swap >> swap.dagger() == Tensor.id(Dim(2, 3))
assert swap.dagger() >> swap == Tensor.id(Dim(3, 2))
matrix1 = Tensor(list(range(9)), Dim(3), Dim(3))
assert vector @ matrix1 >> swap == matrix1 @ vector

cup, cap = Tensor.cups(Dim(2), Dim(2)), Tensor.caps(Dim(2), Dim(2))
print("cup == {}".format(cup))
print("cap == {}".format(cap))

_id = Tensor.id(Dim(2))
assert cap @ _id >> _id @ cup == _id == _id @ cap >> cup @ _id

print("\n    == ".join(map(str, (cap @ _id >> _id @ cup, _id, _id @ cap >> cup @ _id))))

left_snake = Cap(Dim(2), Dim(2)) @ Id(Dim(2)) >> Id(Dim(2)) @ Cup(Dim(2), Dim(2))
right_snake = Id(Dim(2)) @ Cap(Dim(2), Dim(2)) >> Cup(Dim(2), Dim(2)) @ Id(Dim(2))

Equation(left_snake, Id(Dim(2)), right_snake).draw(figsize=(5, 2), draw_type_labels=False)

_eval = tensor.Functor(
    ob=lambda x: x,
    ar=lambda f: f)
assert _eval(left_snake) == _eval(Id(Dim(2))) == _eval(right_snake)

f = tensor.Box("f", Dim(2), Dim(2), data=[1, 2, 3, 4])
Equation(f.transpose(), f.r).draw(figsize=(3, 2), draw_type_labels=False)
assert f.r.eval() == f.transpose().eval()
print(f.r.eval())

%timeit f.transpose().transpose().eval()