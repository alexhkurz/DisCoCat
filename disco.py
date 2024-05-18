from discopy.grammar.pregroup import Ty, Id, Word, Cup, Diagram
from discopy.cat import Category
from discopy.tensor import Dim, Tensor
from discopy import tensor
from discopy.drawing import Equation
from discopy.quantum import circuit, qubit, sqrt, X, Ket, H, CX

n, s = Ty('n'), Ty('s')
Alice = Word("Alice", n)
loves = Word("loves", n.r @ s @ n.l)
Bob = Word("Bob", n)
grammar = Cup(n, n.r) @ s @ Cup(n.l, n)
sentence = Alice @ loves @ Bob >> grammar
sentence.draw(figsize=(5, 5))

F = tensor.Functor(
    ob={n: 2, s: 1},
    ar={Alice: [0, 1], loves: [0, 1, 1, 0], Bob: [1, 0]},
    dom=Category(Ty, Diagram))
print(F(Alice @ loves @ Bob))
print(F(grammar))
assert F(Alice @ loves @ Bob >> grammar).array == 1

assert not F(Alice) >> F(Bob).dagger()
Equation(Alice, Bob, symbol="$\\neq$").draw(figsize=(3, 1))

rich, man = Word("rich", n @ n.l), Word("man", n)
F.ar[rich], F.ar[man] = [1, 0, 0, 0], [1, 0]
rich_man = rich @ man >> Id(n) @ Cup(n.l, n)
assert F(Bob) >> F(rich_man).dagger()  # i.e. Bob is a rich man.
Equation(Bob, rich_man).draw(figsize=(5, 2))

q = Ty('q')
Who = Word("Who", q @ s.l @ n)
F.ob[q], F.ar[Who] = 2, [1, 0, 0, 1]
question = Who @ loves @ Bob\
    >> Id(q @ s.l) @ Cup(n, n.r) @ Id(s) @ Cup(n.l, n)\
    >> Id(q) @ Cup(s.l, s)
answer = Alice
assert F(question) == F(answer)
Equation(question, answer).draw(figsize=(6, 3))

F_ = circuit.Functor(
    ob={s: qubit ** 0, n: qubit ** 1},
    ar={Alice: Ket(0),
        loves: sqrt(2) @ Ket(0, 0) >> H @ X >> CX,
        Bob: Ket(1)})
F_.dom = Category(Ty, Diagram)
F_(sentence).draw(figsize=(6, 6))
assert F_(sentence).eval().is_close(F(sentence).cast(complex))
