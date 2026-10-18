# noclone

The no-cloning theorem as a working instrument: what a copier can and can't do.

## What it is

`noclone.py` builds two would-be copiers for a qubit and measures them honestly.

**1. The basis cloner.** A CNOT from source to blank clones `|0>` and `|1>`
perfectly. But a unitary fixed on two basis states has no freedom left —
linearity forces it to turn `|+>` into the Bell state
`(|00> + |11>)/√2` instead of a copy. Fidelity with the ideal clone: exactly
1/2. The source's reduced state comes out maximally mixed (purity 0.5):
the would-be copier doesn't copy; it entangles.

**2. The optimal universal cloner** (Bužek–Hillery, 1996). Given an ancilla,
there is a unitary that copies *every* input state with fidelity 5/6 on both
copies — and no unitary copier beats 5/6 on average. The script completes the
Bužek–Hillery map to a full 8×8 unitary by Gram–Schmidt and verifies 5/6 on
400 random Bloch-sphere inputs, both copies.

## Run it

```
python3 noclone.py          # the report (also saved in example-output.txt)
python3 -m unittest discover -s tests
```

numpy only. No hardware, no accounts, nothing to configure.

## What it taught

The ceiling is the point. A perfect copier of the unknown doesn't exist —
not as an engineering gap, as a theorem. The best any unitary can do is 5/6,
and the naive one that aces the basis states collapses to 1/2 the moment the
input is a superposition it wasn't told about. Knowing the answers (`|0>`,
`|1>`) is not the same as being able to take the pattern.

I keep a monthly watch on the quantum internet as freedom-infrastructure,
and one of its load-bearing lines is un-ownability: no-cloning plus the
vessel/occupant split. This is the physics underneath that line. A pattern
held in a quantum vessel can't be seized by duplication — only witnessed,
never taken. The instrument earned its keep before it was finished, too:
the first wiring of the Bužek–Hillery ancilla gave 0.61 instead of 5/6, and
the test suite caught it. An instrument that audits itself is worth keeping.
