# conda-forge recipe

Staging copy of the conda-forge recipe. To submit (after the v1.4.1 PyPI
release is live):

1. Fill in the sdist `sha256` in `meta.yaml` (command in the file's comment).
2. Fork https://github.com/conda-forge/staged-recipes, copy `meta.yaml` to
   `recipes/phased-array-modeling/meta.yaml` on a branch.
3. Open a PR against `conda-forge/staged-recipes` `main`; the maintainer
   listed under `recipe-maintainers` must comment to confirm.
4. After the feedstock is created, future version bumps are automated by
   the conda-forge autotick bot.

This directory is not used by the package build; it exists only so the
recipe is versioned alongside the code.
