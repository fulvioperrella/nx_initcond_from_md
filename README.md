# nx_initcond_from_md
This program generates initial conditions for Newton-X 26 QM:MM dynamics from PBC MD of a chromophore in solution.
It accepts an input file in YAML format as argument (see example input.yaml file):

```
$ python nx_initcond_from_md.py input.yaml
```

The program generates a TRAJECTORIES folder, with TRAJi subfolders, each constituting a NX ground state QM:MM trajectory input with G16. The `gaussian.com` file in `JOB_AD` is an ONIOM G16 input, featuring AMBER/GAFF2 atom types, atomic charges, ONIOM partition and MM parameters (just non-bonded for the solute, bonded and non-bonded for the solvent).

Required packages: pyyaml, numpy, scipy, mdanalysis
