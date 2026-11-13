# External course-support repositories

The tracked entries below are Git **gitlinks**, not bundled source directories. This repository has no `.gitmodules` file, so `git submodule update --init` is insufficient.

| Entry | Upstream | Recorded commit |
| --- | --- | --- |
| `patrec-labs` | [slp-ntua/patrec-labs](https://github.com/slp-ntua/patrec-labs) | `180ec709c2e26860d05f59106e5cb3be64612d06` |
| `prep-lab` | [slp-ntua/prep-lab](https://github.com/slp-ntua/prep-lab) | `d32cec7e6f06a02f9e5791cd5701c2148a62658f` |

For a local rerun, recover the helpers explicitly from the repository root, provided the destination directories are empty:

```bash
git clone https://github.com/slp-ntua/patrec-labs.git lab_material/patrec-labs
git -C lab_material/patrec-labs checkout 180ec709c2e26860d05f59106e5cb3be64612d06
git clone https://github.com/slp-ntua/prep-lab.git lab_material/prep-lab
git -C lab_material/prep-lab checkout d32cec7e6f06a02f9e5791cd5701c2148a62658f
```

Lab 2 imports modules from `patrec-labs/lab3`; update the path in your notebook copy if using this local location. The recorded commit identifies the support reference in Git; the original notebook cloned an unpinned upstream branch, so this is not proof of its exact executed helper version. Read upstream requirements before installing the helper code.
