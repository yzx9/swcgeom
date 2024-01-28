# Changelog

## **0.15.0**&emsp;<sub><sup>2024-01-28 ([d300b41...8c9e0ab](https://github.com/yzx9/swcgeom/compare/d300b41689ee892fe5a23dad57f92e1054c0f9e9...8c9e0ab45c447d3d8121b0cc4178be5f65d18a56?diff=split))</sup></sub>

### Features

##### `core`

- expose subtree id mapping ([35738a3](https://github.com/yzx9/swcgeom/commit/35738a31b2373a43ab29550335a457c51b7befce))

##### `images`

- add dtype support ([d300b41](https://github.com/yzx9/swcgeom/commit/d300b41689ee892fe5a23dad57f92e1054c0f9e9))

##### `transforms`

- generate image stack patch ([decd3a1](https://github.com/yzx9/swcgeom/commit/decd3a1ee63e1657aafa24ec5dfbf7439d25a551))
- add \`extra\_repr\` api ([ca19361](https://github.com/yzx9/swcgeom/commit/ca1936146783ebccc2ffe25596799580e89b3432))

### Bug Fixes

##### `core`

- fix classmethod typing annotations ([0123eeb](https://github.com/yzx9/swcgeom/commit/0123eeb46142c3c3a695a5cf2495588a2effe275))

##### `transforms`

- should stack images ([8c9e0ab](https://github.com/yzx9/swcgeom/commit/8c9e0ab45c447d3d8121b0cc4178be5f65d18a56))


### BREAKING CHANGES
- `core` remove assembler module ([414533d](https://github.com/yzx9/swcgeom/commit/414533d17238ed45972dff1909cf025a19b7fa1e))
<br>


## **0.14.0**&emsp;<sub><sup>2023-12-26 ([b9c95f5...cc22018](https://github.com/yzx9/swcgeom/compare/b9c95f58f725490f3b624233cc0be20b31605e53...cc22018ad79af18ecd812b932812a0ff40a0fe40?diff=split))</sup></sub>

### Features

##### `analysis`
* add high accuracy volume calculation ([9b51b74](https://github.com/yzx9/swcgeom/commit/9b51b7401efffc1b0cfbfecbdf3eb60300e1e114))

### Bug Fixes

##### `analysis`
* subtract volume between frustum cone ([dae116e](https://github.com/yzx9/swcgeom/commit/dae116e45146428f5cb61c696ff1d6804e03993c))

##### `utils`
* avoid numerical error ([a6bd4ad](https://github.com/yzx9/swcgeom/commit/a6bd4adc87e86dcaca2bcb45aee68cea51a540c0))

### Performance Improvements

##### `analysis`
* use bvh scene ([7a7b988](https://github.com/yzx9/swcgeom/commit/7a7b98864fd6d0378b0b696d2963353d0592b4d3))

##### `utils`
* use \`sdflit\` instead custom impl ([a47a190](https://github.com/yzx9/swcgeom/commit/a47a19091b46a1071140a12fb20105cdf851a85a))

### BREAKING CHANGES
* `transforms` rewrite image stack ([cc22018](https://github.com/yzx9/swcgeom/commit/cc22018ad79af18ecd812b932812a0ff40a0fe40))

<br>


## **0.13.2**&emsp;<sub><sup>2023-12-16 ([10f330d...291ae2a](https://github.com/yzx9/swcgeom/compare/10f330d13b49aba19c043b91c42c183e1f9ad4d4...291ae2afde6d29d234e8f1b3521d3f221350e01e?diff=split))</sup></sub>

### Performance Improvements
* impl a direct algorithm for finding sphere\-line intersections ([410acad](https://github.com/yzx9/swcgeom/commit/410acad545df66d69f8f04d693310986e53f17f6))

<br>


## **0.13.1**&emsp;<sub><sup>2023-12-15 ([0b5b45a...45b4040](https://github.com/yzx9/swcgeom/compare/0b5b45a7997a95298075635d32552dcd91002e4a...45b4040394b3d6a5636d069b5442945d9134b428?diff=split))</sup></sub>

### Bug Fixes
* import missing volume related file ([0b5b45a](https://github.com/yzx9/swcgeom/commit/0b5b45a7997a95298075635d32552dcd91002e4a))

##### `analysis`
* add volume between spheres ([b8c6dfe](https://github.com/yzx9/swcgeom/commit/b8c6dfe5d6cf53f35dbe8fccfb47d19a2b802753))

##### `utils`
* remove debug printting ([dd1569e](https://github.com/yzx9/swcgeom/commit/dd1569e5a02fef185050e2640ba583b3da60e3db))
* solve real values only ([2f7459b](https://github.com/yzx9/swcgeom/commit/2f7459b06b6a1f3b284f7bbec688ac871d232d1a))

<br>


## **0.13.0**&emsp;<sub><sup>2023-12-14 ([e2add59...06239bd](https://github.com/yzx9/swcgeom/compare/e2add59652bfc02d802f6770ea2c5fbc3fd7d729...06239bd129e6fab329ec80326352b48049fb504e?diff=split))</sup></sub>

### Features

##### `analysis`
* import sholl plot ([b03b45c](https://github.com/yzx9/swcgeom/commit/b03b45c4f20f2ed57263b1c2398316533b45b837))
* calc volume of tree \(close \#9\) ([a5004da](https://github.com/yzx9/swcgeom/commit/a5004dab71e71e68fd4a512757c9310f557878cf))

##### `core`
* check if it has a cyclic \(\#1\) ([e2add59](https://github.com/yzx9/swcgeom/commit/e2add59652bfc02d802f6770ea2c5fbc3fd7d729))

##### `utils`
* transform batch of vec3 to vec4 ([d2d660c](https://github.com/yzx9/swcgeom/commit/d2d660ca53b9886a81b02193ea77f76da4620ffa))

### Bug Fixes

##### `utils`
* should support \`StringIO\` ([de439db](https://github.com/yzx9/swcgeom/commit/de439dba00ce7407d4ae18c9427eab1e5af4d95e))

### Performance Improvements
* improve dsu ([8b414c3](https://github.com/yzx9/swcgeom/commit/8b414c37f4fc3f4ded9c8b19eb8ee0ad52dedd53))

<br>


## **0.12.0**&emsp;<sub><sup>2023-10-12 ([d9ba943...0824e9b](https://github.com/yzx9/swcgeom/compare/d9ba9433735c69edf979013632278e5f498a6fe0...0824e9b4110f820cd11c469ca6e319c1b2f14145?diff=split))</sup></sub>

### Features

##### `core`

* support read from io ([8fe9df8](https://github.com/yzx9/swcgeom/commit/8fe9df8459e8cef3cd6bde4ff3546e1a83871eda))
* get neurites and dendrites ([a9acfde](https://github.com/yzx9/swcgeom/commit/a9acfde5ab77bac22d36e7c461089f5159e6330a))
* add swc types ([7439288](https://github.com/yzx9/swcgeom/commit/7439288d199d558cd170600b07d1ab51a8489bc4))
* add type check in \`Tree\.Node\.is\_soma\` ([35b53d6](https://github.com/yzx9/swcgeom/commit/35b53d68c30e444b0b8ae57519f8196c5dbdcb4d))

##### `images`

* support \`v3dpbd\` and \`v3draw\` \(close \#6\) ([ca8267d](https://github.com/yzx9/swcgeom/commit/ca8267d694f62abc59b9c8174efc91512a9ccec9))

##### `transforms`

* sort mst tree by default ([7878f3f](https://github.com/yzx9/swcgeom/commit/7878f3fdb9beeebd31ecfa7649611fdd03e34eff))
* add path transforms ([aaa1b1e](https://github.com/yzx9/swcgeom/commit/aaa1b1e3c718017ae18a52d91918b29c02f302b3))

##### `utils`

* change to utf\-8 encode ([45e971e](https://github.com/yzx9/swcgeom/commit/45e971eef9cc88abb8e1fca5f26b99e46fcb5aaf))

### Bug Fixes

##### `core`

* avoid duplicate cols comments ([f99eaf3](https://github.com/yzx9/swcgeom/commit/f99eaf3946846522d7eac1769f6a9a4fd20e8bf0))
* inherit source ([702efab](https://github.com/yzx9/swcgeom/commit/702efabb5db9a27e8eaed9d1feb45f24e6b4e808))
* remove original point when cat tree ([065125e](https://github.com/yzx9/swcgeom/commit/065125e7ccf0d3f6ef9ff9c7689a0e7aeb6be349))

##### `images`

* shape should be vec4i ([0824e9b](https://github.com/yzx9/swcgeom/commit/0824e9b4110f820cd11c469ca6e319c1b2f14145))

##### `transforms`

* add missing exports ([93bf8e6](https://github.com/yzx9/swcgeom/commit/93bf8e6076d8d7a3898a1ff88cbe05d855d1173c))

### Performance Improvements

##### `transforms`

* disable \`detach\` operation ([204d44c](https://github.com/yzx9/swcgeom/commit/204d44cbab563309d85f9c6f64793e5eda028547))

<br>

## **0.11.1**&emsp;<sub><sup>2023-08-12 ([36fa413...36fa413](https://github.com/yzx9/swcgeom/compare/36fa4135f2001694b8caea74e82c5ffa1118e90d...36fa4135f2001694b8caea74e82c5ffa1118e90d?diff=split))</sup></sub>

*no relevant changes*

<br>

## **0.11.0**&emsp;<sub><sup>2023-08-05 ([a8adae5...c075ec4](https://github.com/yzx9/swcgeom/compare/a8adae5d58392bcac737a28bdf6c510d07e30fe6...c075ec4063d387d35dcfdd0a021a3407ccd67770?diff=split))</sup></sub>

### Features

##### `analysis`

* draw trunk ([4d2e069](https://github.com/yzx9/swcgeom/commit/4d2e069a86a105c585425d730824c8f22a293d01))
* draw point at the start of subtree ([2c9938f](https://github.com/yzx9/swcgeom/commit/2c9938f9c75aaac807b136baa0f7bd0d8fc4af0a))

##### `core`

* detect swc encoding ([06087bd](https://github.com/yzx9/swcgeom/commit/06087bde32f68b559d0591479538dbf960b92bcd))
* preserve original swc comments\(close \#2\) ([ac49e25](https://github.com/yzx9/swcgeom/commit/ac49e253cf5a33a27a7e61c84b821d944307b857))
* accept non\-positive radius ([47c29b5](https://github.com/yzx9/swcgeom/commit/47c29b5503b0ff41ce678491d22ae715383fb2ee))
* accept space before comments in swc ([c075ec4](https://github.com/yzx9/swcgeom/commit/c075ec4063d387d35dcfdd0a021a3407ccd67770))

##### `utils`

* add neuromorpho related util ([4a22e3d](https://github.com/yzx9/swcgeom/commit/4a22e3d4b8ddf2265fe23106751c0e25232babef))
* convert neuromorpho lmdb to swc ([e48c1a8](https://github.com/yzx9/swcgeom/commit/e48c1a8b7e1397ae17921f36471fbc4ba1c2cff2))
* retry download neuromorpho ([ebd9255](https://github.com/yzx9/swcgeom/commit/ebd92557effe525d90c26f32dfa773fc4977f212))
* mark invalid neuromorpho data ([43a96c6](https://github.com/yzx9/swcgeom/commit/43a96c6e0872c82d4df10783687208ab959f7a11))

### Bug Fixes

##### `core`

* forward names ([52192ce](https://github.com/yzx9/swcgeom/commit/52192ceaa9b687fddb98c3767412aee6cc226ad8))

##### `transform`

* cut by dynamic type ([3293a67](https://github.com/yzx9/swcgeom/commit/3293a675830200c73761c48bc1990554f7d0c82c))

##### `utils`

* should throw http error ([7fb4ee2](https://github.com/yzx9/swcgeom/commit/7fb4ee2ca0ca4308c7d527bec9eb325d22ca0df7))

<br>

## **0.10.0**&emsp;<sub><sup>2023-05-23 ([bc48787...8991fe0](https://github.com/yzx9/swcgeom/compare/bc487879ac6655de0a86504d24a2f67ba6afe848...8991fe0ca647febccba7a9bce7facf0e06334a5c?diff=split))</sup></sub>

### Features

* dict\-like swc ([a1f52d7](https://github.com/yzx9/swcgeom/commit/a1f52d7ac3c03df6077fd038fd8833d1fef03c8e))

##### `analysis`

* add str input support ([f98aea5](https://github.com/yzx9/swcgeom/commit/f98aea59b5ab71196691423185d6a2f896c352c3))

##### `core`

* check if is sorted topology ([bc48787](https://github.com/yzx9/swcgeom/commit/bc487879ac6655de0a86504d24a2f67ba6afe848))

##### `images`

* detect tiff axes ([b1e44bb](https://github.com/yzx9/swcgeom/commit/b1e44bb8c35c299fdb5e4b8b3be61358d59ee7b6))
* change terafly to a left\-handed coordinate system ([b22b69a](https://github.com/yzx9/swcgeom/commit/b22b69acf7715388b0f818eb9c176bee6f39e60a))

##### `transforms`

* cut tree by type ([184482b](https://github.com/yzx9/swcgeom/commit/184482bbb5803e429ac788fa315c0638ad233754))
* reset radius ([e697e34](https://github.com/yzx9/swcgeom/commit/e697e34c3532dcaf6c9b2c1c8afd820d127f3303))

### Bug Fixes

##### `*`

* np\.nonzero returns a tuple ([091e6eb](https://github.com/yzx9/swcgeom/commit/091e6eb84a464e7ab3978ad74de891fa31f98803))

##### `core`

* forwarding init kwargs ([226b3ef](https://github.com/yzx9/swcgeom/commit/226b3efb32cdb6d421bc9aca5d6a5b2f0f0b6e3c))

##### `transforms`

* crop fixed shape ([e1078d3](https://github.com/yzx9/swcgeom/commit/e1078d33414ef02e27b93b9149bf776a6334b1ca))

### Performance Improvements

##### `transforms`

* flat transforms ([f824651](https://github.com/yzx9/swcgeom/commit/f824651e202015a29716dd441391bbb28f0a3bfa))

### BREAKING CHANGES

* `*` export common classes and methods only ([39de173](https://github.com/yzx9/swcgeom/commit/39de173fb3df8967a7edf46b269c216e73f2cb41))
* `core` set \`check\_same\` arg to false by default ([0f576e9](https://github.com/yzx9/swcgeom/commit/0f576e9ba352138d4d138a15bd6f12ea3a8e48db))
* `images` change terafly to a left\-handed coordinate system ([b22b69a](https://github.com/yzx9/swcgeom/commit/b22b69acf7715388b0f818eb9c176bee6f39e60a))

<br>

## **0.9.0**&emsp;<sub><sup>2023-05-04 ([1a3e9d4...797934f](https://github.com/yzx9/swcgeom/compare/1a3e9d4f7a0d1442fbb62f4693f5b5ec6d787af4...797934f120c0de266fc7d21038b5d17a1e2fbabc?diff=split))</sup></sub>

### Features

##### `analysis`

- plot sholl circles by default ([797934f](https://github.com/yzx9/swcgeom/commit/797934f120c0de266fc7d21038b5d17a1e2fbabc))

##### `images`

- add transform support to folder ([167f67c](https://github.com/yzx9/swcgeom/commit/167f67c6f6ba58e80280d66df27caf47cf3dd3e6))
- support disable compression ([9db01e9](https://github.com/yzx9/swcgeom/commit/9db01e94b063c1ec9efcf8fb906f2c732805e45f))
- check if is a valid fname ([4d98abc](https://github.com/yzx9/swcgeom/commit/4d98abc66955d62760047c4fcfb1007f1dc9d64a))

##### `transform`

- add image stack center transform ([0fd1441](https://github.com/yzx9/swcgeom/commit/0fd14411e7b552f7b6d66a1f9c77f69adb0c379a))
- expose static method ([c65c0cb](https://github.com/yzx9/swcgeom/commit/c65c0cba316797bedda5070b2db2a1a622931c20))

### Bug Fixes

##### `core`
- change \`w\` to ones ([325ff1e](https://github.com/yzx9/swcgeom/commit/325ff1efe5cc662f7731003919f313cdd219caad))

### BREAKING CHANGES

- `analysis` plot sholl circles by default ([797934f](https://github.com/yzx9/swcgeom/commit/797934f120c0de266fc7d21038b5d17a1e2fbabc))
- `images` folder do not move channel to first ([fd3a3cb](https://github.com/yzx9/swcgeom/commit/fd3a3cbea59f910ab31682c9272957cfea56eaa4))

<br>

## **0.8.0**&emsp;<sub><sup>2023-04-18 ([3d2d660...bfee570](https://github.com/yzx9/swcgeom/compare/3d2d66034b269b1acf9a62cfeb8a8d3b85d1791a...bfee570d9989f10e5784323a60d2f06d5cf0877d?diff=split))</sup></sub>

### Features

##### `analysis`

- add node count ([149ecec](https://github.com/yzx9/swcgeom/commit/149ecec3543d9f5f63b5a140deb7975328f08589))
- transform each tree in population ([fcd5fd7](https://github.com/yzx9/swcgeom/commit/fcd5fd734712a530548ef4c03e152045156281b3))
- plot sholl in populations ([466cd3a](https://github.com/yzx9/swcgeom/commit/466cd3a8e1a85f68945f01fe862b039677da5d95))

##### `core`

- check is it a binary tree ([b919de9](https://github.com/yzx9/swcgeom/commit/b919de98654e3ed5b0934c3757ed22f6024c563c))
- support create population from trees ([73792a0](https://github.com/yzx9/swcgeom/commit/73792a03f4b69ea3854f5f4dc3226f2301641a58))
- check valid population ([199db95](https://github.com/yzx9/swcgeom/commit/199db957d73c2eb44ef538f85f97dca127d5a339))

##### `images`

- add image stack folder ([0318f0d](https://github.com/yzx9/swcgeom/commit/0318f0d9fc7ea513b0f9c24db1f7e3254d0ffbbb))
- pathed image stack folder ([bfee570](https://github.com/yzx9/swcgeom/commit/bfee570d9989f10e5784323a60d2f06d5cf0877d))

##### `transforms`

- add \`LinesToTree\` ([5083f26](https://github.com/yzx9/swcgeom/commit/5083f2631935cfec3db0ad5d2e091e3849e14294))
- add branch smoother ([a4e8cd2](https://github.com/yzx9/swcgeom/commit/a4e8cd2636fafd1d7cec8a650b54f72c37c78250))
- add tree smoother ([082b05d](https://github.com/yzx9/swcgeom/commit/082b05df466cd83cfb6606b804d12192220602eb))
- add transform \`TranslateOorigin\` ([5ef9c43](https://github.com/yzx9/swcgeom/commit/5ef9c43703cf08c0ea3074b1dda6d514f7daed95))
- expose \`AffineTransform\` ([93d6ca9](https://github.com/yzx9/swcgeom/commit/93d6ca947cf9745a59af44e60e975de7269c8f78))

##### `transfroms`

- get tree longest path ([17b6eda](https://github.com/yzx9/swcgeom/commit/17b6eda9adc4aa32b721ac5084ba0906b2e79a7f))

### Bug Fixes

##### `analysis`

- handle different number of trees ([94ffd2e](https://github.com/yzx9/swcgeom/commit/94ffd2ee3cc48beef295a1d60d31400a6b400bbc))
- wrong x\-ticks ([ebd2e91](https://github.com/yzx9/swcgeom/commit/ebd2e911aafed1b546684d4832d453f8d0489b48))

##### `core`

- set new root ([54cecf2](https://github.com/yzx9/swcgeom/commit/54cecf2bcec9473fa5141eccf0b254428dc229a9))
- detach with new id and pid ([a5f1791](https://github.com/yzx9/swcgeom/commit/a5f17914fe59a6695ce2508ca7a0626500557f19))

##### `images`

- patch has 4 dim ([ccdb4d7](https://github.com/yzx9/swcgeom/commit/ccdb4d7c174bb408de7187be092e2af66033808e))

##### `transform`

- typo ([7bfa106](https://github.com/yzx9/swcgeom/commit/7bfa10688c3cf437339cbc4d9a60a34d95d986a9))

### Performance Improvements

##### `images`

- add lru cache for terafly ([f288710](https://github.com/yzx9/swcgeom/commit/f288710f85c5564060a89b7de70cd5af195a9dc1))

<br>

## **0.7.0**&emsp;<sub><sup>2023-04-02 ([e45b520...df0b693](https://github.com/yzx9/swcgeom/compare/e45b520fc841d4d8ec8d2f49f2505aa099cba730...df0b693facd584fb0c928126256d067fbf896ea3?diff=split))</sup></sub>

### Features

##### `core`

- update root orf tree ([13395d4](https://github.com/yzx9/swcgeom/commit/13395d471bec3bb076471923b532f82efee61475))
- concat tree ([93abf8c](https://github.com/yzx9/swcgeom/commit/93abf8c33b97b3fffa42a46fd8c2d921d89d750b))
- check is it an binary tree ([ba1b5b9](https://github.com/yzx9/swcgeom/commit/ba1b5b924f61bcc5121071e8e2a7d5f43dbb8c13))
- remove duplicate nodes ([95596de](https://github.com/yzx9/swcgeom/commit/95596de41bebd301b371bff4f37e07fb3a6a384d))
- update waning message ([f745626](https://github.com/yzx9/swcgeom/commit/f745626eba5a96927622821c9a5d9b8b20943966))

##### `images`

- support rgb ([e45b520](https://github.com/yzx9/swcgeom/commit/e45b520fc841d4d8ec8d2f49f2505aa099cba730))

##### `transforms`

- add mst tree ([529a530](https://github.com/yzx9/swcgeom/commit/529a530f9f2901bf0ade8054b8e8b75392559306))

### BREAKING CHANGES

- `utils` \`utils\.numpy_printoptions\` has been remove, use \`np\.printoptions\`instead of ([df0b693](https://github.com/yzx9/swcgeom/commit/df0b693facd584fb0c928126256d067fbf896ea3))

<br>

## **0.6.0**&emsp;<sub><sup>2023-03-22 ([97cb9b2...d40ccf6](https://github.com/yzx9/swcgeom/compare/97cb9b2aed70237de2a935a48764b0c384cdcbb4...d40ccf65568085f6c802c3558d1e0815d2e280cf?diff=split))</sup></sub>

### Features

##### `analysis`

- improve node_branch_order plotting ([97cb9b2](https://github.com/yzx9/swcgeom/commit/97cb9b2aed70237de2a935a48764b0c384cdcbb4))
- add tip and bifurcation node feature ([c1594d9](https://github.com/yzx9/swcgeom/commit/c1594d907f21b47f06d86c5bdd8e4e3cdbb7fd79))
- show fig by default ([1587488](https://github.com/yzx9/swcgeom/commit/158748890e6c354b07fb3e4f24da2a5bf1e7a416))
- filter tips node ([c3dd567](https://github.com/yzx9/swcgeom/commit/c3dd567a5387dd44f6e035d527aa4883957dec1c))
- rebuild sholl ([a43bc01](https://github.com/yzx9/swcgeom/commit/a43bc01419cf74f87e698b839dc651ae54d68d69))

##### `core`

- traverse from node ([9f64d31](https://github.com/yzx9/swcgeom/commit/9f64d315fbeb01f80624734429404ac7a681d67f))
- support index by np\.integer ([1b1699f](https://github.com/yzx9/swcgeom/commit/1b1699fcf612395b68da9a436beddc593f041628))
- get subtree rooted at n ([b57fb62](https://github.com/yzx9/swcgeom/commit/b57fb6280395064ef0b2a81c38d559e09ed650e6))
- add high\-level subtree api ([eb91d7f](https://github.com/yzx9/swcgeom/commit/eb91d7f9b0ebb5d0f5af99416c5a3818fcfc011d))

##### `images`

- add io of image stack ([f7b27be](https://github.com/yzx9/swcgeom/commit/f7b27bee74b70470f974ecc2197fbf5331abfdd7))

##### `transforms`

- add geometry transforms ([6fd1166](https://github.com/yzx9/swcgeom/commit/6fd1166fc81f774630d9cf431defb995eb07a6df))

<br>

## **0.5.0**&emsp;<sub><sup>2023-03-09 ([711ebb1...80acd75](https://github.com/yzx9/swcgeom/compare/711ebb1308d2a70eddcf6295a40883a7815e887e...80acd75f7a9c232008f5ce4c9ec1f43488efd21c?diff=split))</sup></sub>

### Features

##### `analysis`

- custom draw ([bdc1291](https://github.com/yzx9/swcgeom/commit/bdc1291eb4e213ad3918c70dad54f00d81be6769))
- re\-design feature_extractor ([e5affa7](https://github.com/yzx9/swcgeom/commit/e5affa700128d62e175bb6243ba1d7f00ac27c85))
- add populations feature extractor ([94a2bc6](https://github.com/yzx9/swcgeom/commit/94a2bc69a9eb1856138e54fc858a48480c324c61))
- plot histogram ([0dc05d7](https://github.com/yzx9/swcgeom/commit/0dc05d72a7b1dcff8f2e64e6b79a91e3d040ac8a))
- custom length plot ([d32272d](https://github.com/yzx9/swcgeom/commit/d32272d242274e8bdf2acd4d3dc80e5858481449))
- custom position of direction indicator ([b84b57e](https://github.com/yzx9/swcgeom/commit/b84b57eb912da55b54b4d72e526d4cf7e96abc64))
- support hidden swc in legend ([80acd75](https://github.com/yzx9/swcgeom/commit/80acd75f7a9c232008f5ce4c9ec1f43488efd21c))

##### `core`

- add populations ([711ebb1](https://github.com/yzx9/swcgeom/commit/711ebb1308d2a70eddcf6295a40883a7815e887e))
- rename class method option ([7003c1d](https://github.com/yzx9/swcgeom/commit/7003c1dc017a0ca432265c003434e871ea9c1fc6))
- take intersection among populations ([6442572](https://github.com/yzx9/swcgeom/commit/6442572c9fe32621dd93b6288d82e40a45e6012f))
- add pure\-func style utils ([7f9cba1](https://github.com/yzx9/swcgeom/commit/7f9cba14dd4bc0e018fe03bccd1cf845d4a3a421))
- add \`BranchTree\.from_eswc\` ([d01cd04](https://github.com/yzx9/swcgeom/commit/d01cd04505649cc782edbc34d5ac7205093b57de))
- add labels for populations ([998566e](https://github.com/yzx9/swcgeom/commit/998566e4c478716ad57a77b52dc6774608d4e797))

### Merges

- branch 'feature/populations' ([1c7d1e8](https://github.com/yzx9/swcgeom/commit/1c7d1e87a6a34ab79146adfe9e5e95b24f636432))

### BREAKING CHANGES

- `analysis` ([e5affa7](https://github.com/yzx9/swcgeom/commit/e5affa700128d62e175bb6243ba1d7f00ac27c85))
  - \`extract_feature\` has been fully re\-designed
  - \`NodeFeatures\.get_distribution\` has been removed
  - \`Sholl\.get_distribution\` has been removed
- `analysis` remove filters in \`NodeFeatures\` ([f0b37bf](https://github.com/yzx9/swcgeom/commit/f0b37bf075fef80a71442d75ae84132796f52c28))
- `core` \`Population.from_swc\` and \`Population\.from_eswc\` argument \`suffix\` has been renamed to \`ext\` ([7003c1d](https://github.com/yzx9/swcgeom/commit/7003c1dc017a0ca432265c003434e871ea9c1fc6))
- `core` add pure\-func style utils ([7f9cba1](https://github.com/yzx9/swcgeom/commit/7f9cba14dd4bc0e018fe03bccd1cf845d4a3a421))

<br>

## **0.4.1** <sub><sup>2023-03-01 ([4efbcef...4efbcef](https://github.com/yzx9/swcgeom/compare/4efbcef...4efbcef?diff=split))</sup></sub>

### Bug Fixes

##### `core`

- should count nonzero nodes ([4efbcef](https://github.com/yzx9/swcgeom/commit/4efbcef))

<br>

## **0.4.0** <sub><sup>2023-02-27 ([af51d11...1d9aeb5](https://github.com/yzx9/swcgeom/compare/af51d11...1d9aeb5?diff=split))</sup></sub>

### Features

##### `analysis`

- forward kwarg to plot method ([1d9aeb5](https://github.com/yzx9/swcgeom/commit/1d9aeb5))

##### `core`

- add \`Tree\.from_data_frame\` ([b36f382](https://github.com/yzx9/swcgeom/commit/b36f382))
- assemble lines into swc ([382df7b](https://github.com/yzx9/swcgeom/commit/382df7b))
- sort nodes after assemble ([d00687c](https://github.com/yzx9/swcgeom/commit/d00687c))
- rename \`sort_swc\` to \`sort_nodes\` ([b7f27d6](https://github.com/yzx9/swcgeom/commit/b7f27d6))
- support export swc without source ([a8da734](https://github.com/yzx9/swcgeom/commit/a8da734))

##### `transform`

- support cut short branch ([0bc2f2c](https://github.com/yzx9/swcgeom/commit/0bc2f2c))

### Performance Improvements

##### `core`

- reduce the number of value fetches ([33169d1](https://github.com/yzx9/swcgeom/commit/33169d1))

### BREAKING CHANGES

- `analysis` forward kwarg to plot method ([1d9aeb5](https://github.com/yzx9/swcgeom/commit/1d9aeb5))

<br>

## **0.3.1** <sub><sup>2023-02-13 ([557693b...fc9335f](https://github.com/yzx9/swcgeom/compare/557693b...fc9335f?diff=split))</sup></sub>

_no relevant changes_

<br>

## **0.3.0** <sub><sup>2023-02-12 ([40ff824...cedf52f](https://github.com/yzx9/swcgeom/compare/40ff824...cedf52f?diff=split))</sup></sub>

### Features

- add tree cutter ([e8f1e68](https://github.com/yzx9/swcgeom/commit/e8f1e68))
- support smart color swither ([a6df232](https://github.com/yzx9/swcgeom/commit/a6df232))
- add debug helper ([f3819a5](https://github.com/yzx9/swcgeom/commit/f3819a5))
- sort tree ([5f1d53e](https://github.com/yzx9/swcgeom/commit/5f1d53e))
- support to swc string ([0b68b65](https://github.com/yzx9/swcgeom/commit/0b68b65))
- sort swc tree ([d812c0d](https://github.com/yzx9/swcgeom/commit/d812c0d))

##### `analysis`

- add get distribution ([ea44851](https://github.com/yzx9/swcgeom/commit/ea44851))
- add population feature extractor ([7f3ff59](https://github.com/yzx9/swcgeom/commit/7f3ff59))
- plot feature distribution ([460f7d1](https://github.com/yzx9/swcgeom/commit/460f7d1))
- support plot sholl ([a253d60](https://github.com/yzx9/swcgeom/commit/a253d60))
- expose plot ax ([9bb7585](https://github.com/yzx9/swcgeom/commit/9bb7585))

##### `analysis/feature`

- expose ax ([2f98884](https://github.com/yzx9/swcgeom/commit/2f98884))
- minor ([bf18b4d](https://github.com/yzx9/swcgeom/commit/bf18b4d))
- change default options ([c6891b0](https://github.com/yzx9/swcgeom/commit/c6891b0))

##### `analysis/visualization`

- add legend ([3739d93](https://github.com/yzx9/swcgeom/commit/3739d93))

##### `core`

- add suffix option ([ad8b1b0](https://github.com/yzx9/swcgeom/commit/ad8b1b0))_ rename \`get_branch\` to \`branch\` ([7adca54](https://github.com/yzx9/swcgeom/commit/7adca54))_ export eswc ([0bce6a8](https://github.com/yzx9/swcgeom/commit/0bce6a8))

##### `core/swc`

- write 1 as root id by default ([946084b](https://github.com/yzx9/swcgeom/commit/946084b))
- support eswc ([f49906a](https://github.com/yzx9/swcgeom/commit/f49906a))
- fix multi roots ([de75ee2](https://github.com/yzx9/swcgeom/commit/de75ee2))
- sort tree ([a7ac4c7](https://github.com/yzx9/swcgeom/commit/a7ac4c7))
- rename read params ([cb55c2b](https://github.com/yzx9/swcgeom/commit/cb55c2b))

##### `core/tree`

- expose ndata id ([0464b3f](https://github.com/yzx9/swcgeom/commit/0464b3f))

##### `datasets`

- rename \`TreeToBranchTree\` to \`ToBranchTree\` ([5f6c93c](https://github.com/yzx9/swcgeom/commit/5f6c93c))

##### `transform`

- rename option ([ced31b1](https://github.com/yzx9/swcgeom/commit/ced31b1))
- add image stack ([114f119](https://github.com/yzx9/swcgeom/commit/114f119))
- add verbose option ([31a32a2](https://github.com/yzx9/swcgeom/commit/31a32a2))
- improve perf ([40c650d](https://github.com/yzx9/swcgeom/commit/40c650d))

##### `transform/image-stack`

- transforms population ([a686fb1](https://github.com/yzx9/swcgeom/commit/a686fb1))

##### `transform/image_stack`

- support big image stack ([bbe5bd1](https://github.com/yzx9/swcgeom/commit/bbe5bd1))
- add photometric ([35e40c5](https://github.com/yzx9/swcgeom/commit/35e40c5))

### Bug Fixes

- should copy ndata ([cedf52f](https://github.com/yzx9/swcgeom/commit/cedf52f))

##### `analysis/visulization`

- handle draw too many neurons in same axes ([de48fe2](https://github.com/yzx9/swcgeom/commit/de48fe2))

##### `core`

- avoid duplicate export tree ([4a506bb](https://github.com/yzx9/swcgeom/commit/4a506bb))
- tip should not be parent ([ce22ea0](https://github.com/yzx9/swcgeom/commit/ce22ea0))
- branch should be nodes ([3f1dcb0](https://github.com/yzx9/swcgeom/commit/3f1dcb0))
- should returns id ([c622666](https://github.com/yzx9/swcgeom/commit/c622666))
- handle zero length ([0112949](https://github.com/yzx9/swcgeom/commit/0112949))
- should propagate removal ([f87ab43](https://github.com/yzx9/swcgeom/commit/f87ab43))

##### `datasets/dgl`

- copy prop ([40ff824](https://github.com/yzx9/swcgeom/commit/40ff824))

##### `transfrom/image_stack`

- handle root sdf ([ce8cae2](https://github.com/yzx9/swcgeom/commit/ce8cae2))

### Performance Improvements

##### `transform/image_stack`

- boost intersect ([8003026](https://github.com/yzx9/swcgeom/commit/8003026))

##### `transforms`

- boost is_in of sdf ([45fee56](https://github.com/yzx9/swcgeom/commit/45fee56))

### BREAKING CHANGES

- `analysis` \`get_distribution\` returns both \`x\` and \`y\`, before ([460f7d1](https://github.com/yzx9/swcgeom/commit/460f7d1))<br>only \`y\`\.
- `anlysis/node_feature` change default options ([c6891b0](https://github.com/yzx9/swcgeom/commit/c6891b0))

<br>

## **0.2.0** <sub><sup>2022-09-28</sup></sub>

### BREAKING CHANGE

- several features have been activatedm, take advantage
- migrate to numpy-based, remove networkx

## **0.1.7** <sub><sup>2022-08-28</sup></sub>

### Features

##### `core`

- expose branch standardize method

##### `data/torch`

- support tree transfroms
- expose `BranchToTensor` transform, and support more options

##### `data/transforms`

- add branch transforms

### Bug Fixes

##### `core`

- stack arrays to get the correct shape
- update node correctly

##### `data/torch`

- super object is not subscriptable

### Performance Improvements

##### `data`

- delay log formatting

<br>

## **0.1.6** <sub><sup>2022-08-25</sup></sub>

### Features

- add `Branch.from_numpy`
- rename `Branch.from_numpy` to `Branch.from_numpy_batch`
- expose branch resampler

### Bug Fixes

##### `data/torch`

- avoid cycle reference

### BREAKING CHANGES

- resample args should be named arguments now
- rename `Branch.from_numpy` to `Branch.from_numpy_batch`, and drop the support
  for squeezed input

<br>

## **0.1.5** <sub><sup>2022-08-24</sup></sub>

### BREAKING CHANGE

- now node attributes are all dictionary attributes

### Features

- convert node to dict

##### `data/torch`

- add datasets

<br>

## **0.1.4** <sub><sup>2022-08-22</sup></sub>

### Bug Fixes

- import version correctly

<br>

## **0.1.3** <sub><sup>2022-08-21</sup></sub>

<br>

## **0.1.2** <sub><sup>2022-08-21</sup></sub>

<br>

## **0.1.1** <sub><sup>2022-08-21</sup></sub>

<br>

## **0.1.0** <sub><sup>2022-08-21</sup></sub>

### Features

- import codes
