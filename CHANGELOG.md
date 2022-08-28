## Unreleased

## Feat

- **data**: add downloader

## v0.1.7 (2022-08-28)

### Feat

- **data/torch**: support tree transfroms
- **data/torch**: expose `BranchToTensor` transform, and support more options
- **data/transforms**: add branch transforms
- **core**: expose branch standardize method

### Fix

- **data/torch**: super object is not subscriptable
- **core**: stack arrays to get the correct shape
- **core**: update node correctly

### Refactor

- **core**: initial dict

### Perf

- **data**: delay log formatting

## v0.1.6 (2022-08-25)

### BREAKING CHANGE

- resample args should be named arguments now
- rename `Branch.from_numpy` to `Branch.from_numpy_batch`, and drop the support
for squeezed input

### Feat

- add `Branch.from_numpy`
- rename `Branch.from_numpy` to `Branch.from_numpy_batch`
- expose branch resampler

### Fix

- **data/torch**: avoid cycle reference

### Refactor

- **\***: rebuild workspace

## v0.1.5 (2022-08-24)

### BREAKING CHANGE

- now node attributes are all dictionary attributes

### Feat

- convert node to dict
- **data/torch**: add datasets

### Refactor

- manually manage node information

## v0.1.4 (2022-08-22)

### Fix

- import version correctly

## v0.1.3 (2022-08-21)

## v0.1.2 (2022-08-21)

## v0.1.1 (2022-08-21)

## v0.1.0 (2022-08-21)

### Feat

- import codes
