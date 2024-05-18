try:
    from discopy import Ty, Word, Cup, Cap, Diagram, Id, Functor
except ImportError:
    print("Cannot import 'Ty' from 'discopy'. Please check your installation.")
from discopy.rigid import Functor

s, n = Ty('s'), Ty('n')  # s for sentence, n for noun

alice = Word("Alice", n)
bob = Word("Bob", n)
loves = Word("loves", n.r @ s @ n.l)

sentence = alice @ loves @ bob >> Id(n) @ Cup(n, n.r) @ Id(n.l)
print("Diagram of the sentence 'Alice loves Bob':")
sentence.draw(figsize=(6, 4), fontsize=12)
