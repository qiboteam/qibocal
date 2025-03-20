# Example platforms

In this folder there are a few example platforms that can be used with Qibocal. To use these platforms `platforms` folder needs to be added to `QIBOLAB_PLATFORMS` environment variable. If the qibocal directory is cloned in the `HOME` directory this can be done with

```bash
export QIBOLAB_PLATFORMS=~/qibocal/platforms
```

If there is more than one directory containing platforms they can be concatenated in this way.

```bash
export QIBOLAB_PLATFORMS=~/qibocal/platforms:<path_to_platforms>:<another_path_to_platforms>
```

This instructions can be added to the configuration file to avoid repeating the instructions for each session.
