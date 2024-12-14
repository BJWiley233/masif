##### Metadynamics


##### Flooding G1
[GLUT1 Structure](https://www.nature.com/articles/nature13306/figures/1)
Ala402-403 are in Helix 11 which crosses with helix 7
Opening PHE-TRP(helix)-glycine-helix is helix 10

Helix-Gly-Helix Sequence Gate-Keeper (Helix 7): 
G1 282-290 (P11166)
QQLSGINAV

Glycine hinges below in helix 7 (occlusion) and helix 10 (opening)


[The Alternating-Access Mechanism of MFS Transporters Arises from Inverted-Topology Repeats](https://www.sciencedirect.com/science/article/pii/S0022283611001410?via%3Dihub)
Helix-Gly-Helix Sequence Opening: 

G1	IC 389 388 387 386 385 384 383 382 381 380 379 EC
G1	   PHE TRP PRO ILE PRO GLY PRO GLY ALA GLU PHE
G3  EC 377 378 379 380 381 382 383 384 385 386 387 IC
G3     PHE GLU ILE GLY PRO GLY PRO ILE PRO TRP PHE

Helix-Gly-Helix Sequence Gate-Keeper: 
G1 282-290 (P11166)
QQLSGINAV

G3 140-130 (P11169)
CGLCTGFVPM
MPVFGTCLGC (inverted)



#### MSMs
PHE flips of 180 degrees in Kinases, 90 degrees in GLUT1??  Any pertainent PHEs in Dioxygenases?

The attribute msm.pi tells us, for each discrete state, the absolute probability of observing said state in global equilibrium. Mathematically speaking, the stationary distribution π
 is the left eigenvector of the transition matrix P
 to the eigenvalue 1
:

$\pi^\top P = \pi^\top$

###### Formula for Timescales
The implied timescales are calculated as:

$\tau_i = -\frac{\tau_{\text{lag}}}{\ln(\lambda_i)}$

where:

- \(i\) is the index of the eigenvalue, with \(i = 2, 3, \dots\) (the first eigenvalue, \(\lambda_1 = 1\), is excluded as it corresponds to the stationary distribution).
  ```python
  ## first 5 for mm are
  mm.eigenvalues()[1:6]
  out: array([0.97855449, 0.8886344 , 0.82838174, 0.75829495, 0.75497462])
  ```
- \(\lambda_i\) is the \(i\)-th eigenvalue of the transition matrix \(P\), computed from the MSM.
- \(\tau_{\text{lag}}\) is the lag time used to estimate the MSM.
