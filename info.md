docker url:

https://hub.docker.com/r/ysa6/genus/

command to build using singularity: 
```
singularity build genus_img.sqsh docker://ysa6/genus:latest
```

command to run the container
```
singularity shell --bind /storage:/storage genus_img.sqsh
```
